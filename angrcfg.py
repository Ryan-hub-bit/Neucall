import angr

import json, random

import os
import time
import csv
import dgl
import torch as th
import numpy

from collections import defaultdict
from dgl.data import DGLDataset
import base64, pickle

import eval_utils as utils


palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")


jsonex='.tgcfi.json' #.ifcc.json
directory = "/home/isec/Documents/experiment_6"
binary_dir = "/home/isec/Documents/experiment_6/valid_binary_list"
json_dir = "/home/isec/Documents/experiment_6/valid_json_list"
txt_dir = "/home/isec/Documents/experiment_6/valid_callsite_txt"
graphdir = "/home/isec/Documents/experiment_6/graph_dir"
addrdir = "/home/isec/Documents/experiment_6/address_dir"
onlyCount = False

codenodeid = 0
datanodeid = 0
funcnodeid = 0
addr_min = 0xffffffff
addr_max = 0
Ninst_addrs = 70  # take the basic blcok  first nth instruction as input
g_list = []
asmdict = {}
naming = 0
logfile = "dsmapping.csv"
logf = open(os.path.join(graphdir, logfile), 'w')

def savebin(filemd5, g, GT_edges, funcsave):
    global naming, logf
    logf.write(str(naming)+','+filemd5+'\n')
    graph_labels = {"GT_label": th.tensor(GT_edges)}
    dgl.data.utils.save_graphs(os.path.join(graphdir, str(naming) + ".graph"), [g], graph_labels)
    with open(os.path.join(graphdir, str(naming) + ".funcaddr"), 'wb') as fp:
        pickle.dump(funcsave, fp)
    naming+=1

# codenode
class codetrainnode():
    def __init__(self, node):
        global codenodeid, palmtree, asmdict, addr_max, addr_min
        self.id = codenodeid
        codenodeid += 1
        self.addr = node.addr
        self.func_addr = node.function_address
        self.text = []
        insns = node.block.capstone.insns
        for insn in insns:
            tmp = insn.mnemonic + ' ' + insn.op_str
            self.text.append(tmp.replace(',','').replace('[','[ ').replace(']',' ]'))
        if len(self.text) == 0:
            self.text = ['nop']
        asmdict[self.addr] = self.text
        self.embeddings = palmtree.encode(self.text)
        self.avg = self.embeddings.mean(axis=0)
        self.embeddings = self.embeddings[:Ninst_addrs]
        addr_max = max(addr_max, self.addr)
        addr_min = min(addr_min, self.addr)

# datanode
class datatrainnode():
    def __init__(self, addr, externfunc_name=None):
        global datanodeid, addr_max, addr_min
        self.id = datanodeid
        datanodeid += 1
        self.addr = addr
        self.externfunc_name = externfunc_name
        addr_max = max(addr_max, self.addr)
        addr_min = min(addr_min, self.addr)


#funcnode
class functrainnode():
    def __init__(self, addr):
        global funcnodeid
        self.id = funcnodeid
        funcnodeid += 1
        self.addr = addr

def obj_dict(obj):
    return obj.__dict__

skipped = 0
total = 0
processed = 0
icalls = 0
icallsite = 0
starttime=time.time()
mlog = open(os.path.join(directory, "output.log"),'w')

# Iterate over the ground truth
# use angr to analyze binary
# construct graph according to the result of angr
for root, dirs, jsonfiles in os.walk(json_dir):
    for binaries in jsonfiles:
        jsonfile = os.path.join(root, binaries)
        print(jsonfile)
        jsonraw = ''
        with open(jsonfile,'r') as f:
            jsonraw = f.read()
            tdict = json.loads(jsonraw)
            #print(tdict)
            binary = binaries[:-len(jsonex)]
            binfile = os.path.join(binary_dir, binary)
            #print(binfile)
            total += 1
            sum_targets = 0
            for calllist in tdict['tg_targets']:
                sum_targets += len(tdict['tg_targets'][calllist])
                if len(tdict['tg_targets'][calllist]) != 0:
                    icallsite +=1
            if sum_targets == 0 or not os.path.isfile(binfile):
                skipped += 1
                continue
            icalls += sum_targets
            processed += 1
            if onlyCount == True:
                continue

            p = angr.Project(binfile, load_options={'auto_load_libs': False})
            cfg = p.analyses.CFGFast(cross_references=True)

            codenodes = []
            datanodes = []

            code2code_edges = []
            codexrefcode_edges = []
            codecall_edges = []
            codexrefdata_edges = []
            dataxrefcode_edges = []
            dataxrefdata_edges = []
            GT_edges = []
            code2func_edges = []

            codenodeid = 0
            datanodeid = 0
            funcnodeid = 0
            addr_min = 0xffffffff
            addr_max = 0

            nodelookup = defaultdict(lambda: None)
            icalllookup = defaultdict(lambda: None)
            symlookup = defaultdict(lambda: None)
            funclookup = defaultdict(lambda: None)

            # name -> addr
            # addr -> name
            for faddr, func in p.kb.functions.items():
                funclookup[func.name] = faddr
                symlookup[faddr] = func.name

            # free is a hook function of angr
            # change the virtual address to rebased_addr
            funcfree = "free"
            a = p.loader.find_symbol(funcfree)
            if a is not None:
                faddr = a.rebased_addr
                funclookup[func] = faddr
                symlookup[faddr] = func

            # icalllookup: {funcaddr -> {ins_addr -> icallee_addr, .....}}
            for _, icall in cfg.indirect_jumps.items():
                if icall.jumpkind == 'Ijk_Call':
                    if icall.func_addr not in icalllookup:
                        newdict = dict()
                        newdict[icall.ins_addr] = icall.addr
                        icalllookup[icall.func_addr] = newdict
                    else:
                        if icall.ins_addr in icalllookup[icall.func_addr]:
                            if icall.addr > icalllookup[icall.func_addr][icall.ins_addr]:
                                icalllookup[icall.func_addr][icall.ins_addr] = icall.addr
                        else:
                            icalllookup[icall.func_addr][icall.ins_addr] = icall.addr

            # Step 1: Code Node
            for node in cfg.graph:
                if node.block is not None:
                    newnode = codetrainnode(node)
                    codenodes.append(newnode)
                    nodelookup[node.addr] = newnode

            # Step 2: Code edge
            for edge in cfg.graph.edges:
                node0 = nodelookup[edge[0].addr]
                node1 = nodelookup[edge[1].addr]
                if node0 is not None and node1 is not None:
                    newedge = (node0.id, node1.id)
                    insns = edge[0].block.capstone.insns
                    if len(insns) == 0:
                        continue
                    if insns[-1].mnemonic == 'call':
                        codecall_edges.append(newedge)
                    else:
                        code2code_edges.append(newedge)

            # Step 3: xrefs edge
            # call functionA
            # find all the reference positon  that call function A, then construct an edge between call_addr -> funcA
            for dst in p.kb.xrefs.xrefs_by_dst:
                xrefs = p.kb.xrefs.xrefs_by_dst[dst]
                for xref in xrefs:
                    node0_addr = xref.block_addr
                    if xref.block_addr is None:
                        node0_addr = xref.ins_addr
                    node0 = nodelookup[node0_addr]
                    node1 = nodelookup[dst]
                    if node0 is None:
                        newnode = datatrainnode(node0_addr, symlookup[node0_addr])
                        datanodes.append(newnode)
                        nodelookup[node0_addr] = newnode
                        node0 = newnode
                    if node1 is None:
                        newnode = datatrainnode(dst, symlookup[dst])
                        datanodes.append(newnode)
                        nodelookup[dst] = newnode
                        node1 = newnode
                    newedge = (node0.id, node1.id)
                    if isinstance(node0, codetrainnode) and isinstance(node1, codetrainnode):
                        codexrefcode_edges.append(newedge)
                    elif isinstance(node0, codetrainnode) and isinstance(node1, datatrainnode):
                        codexrefdata_edges.append(newedge)
                    elif isinstance(node0, datatrainnode) and isinstance(node1, codetrainnode):
                        dataxrefcode_edges.append(newedge)
                    elif isinstance(node0, datatrainnode) and isinstance(node1, datatrainnode):
                        dataxrefdata_edges.append(newedge)

            # Step 4: code 2 func edges
            funcnodelookup = {}
            for node in codenodes:
                if node.func_addr not in funcnodelookup:
                    funcnode = functrainnode(node.func_addr)
                    funcnodelookup[node.func_addr] = funcnode
                else:
                    funcnode = funcnodelookup[node.func_addr]
                code2func_edges.append((node.id, funcnode.id))

            # we store the callsite: address
            # get the address by callsite name
            binaryName = binaries.split(".")[0].split(".")[0]
            def  getCallsiteAddress(callKey):
                callsiteFile = os.path.join(txt_dir, binaryName +".txt")
                if not os.path.exists(callsiteFile):
                    print(f"Err: {callsiteFile} not in {txt_dir}")
                with open(callsiteFile, 'r') as f:
                    callsites =f.readlines()
                callsites = [s[:-1] for s in callsites]
                addressFileName = binaryName + "_addressd.txt"
                addressFile = os.path.join(addrdir, addressFileName)
                if not os.path.exists(addressFile):
                    print(f"Err: {addressFile} not in {addrdir}")
                    return 0
                with open(addressFile,'r') as f:
                    addresses = f.readlines()
                addresses = [s[:-1] for s in addresses]
                addressIndex = callsites.index(callKey)
                callsiteAddress = addresses[addressIndex]
                return callsiteAddress

            # Step 4: add GT info to our graph
            call_dict = tdict['tg_targets']
            for callKey in tdict['tg_targets']:
                callsite_address_str= getCallsiteAddress(callKey)
                #print(f"callsite_address_16: {callsite_address_16}")
                callsite_address = int(callsite_address_str)
                #print(f"callsite_address: {callsite_address}")
                if callsite_address not in nodelookup:
                    print("Err: no callsite in nodelookup")
                    print(callsite_address)
                    continue
                callsite = nodelookup[callsite_address]
                #print(callsite)
                for calltarget in tdict['tg_targets'][callKey]:
                    if calltarget not in funclookup:
                        print("Err:no func in callee")
                        print(calltarget)
                        continue
                    target_addr = funclookup[calltarget]
                    if target_addr is None:
                        continue
                    callee = nodelookup[target_addr]
                    newedge = (callsite.id, callee.id)
                    GT_edges.append(newedge)

            if len(GT_edges) == 0:
                #os.remove(jsonfile)
                continue

            funcsave = {}
            for _, v in funclookup.items():
                if nodelookup[v] is not None:
                    node1 = nodelookup[v]
                    funcsave[v] = node1.id
            # node 有序号  23
            # code func 的节点不是global unique的
            graph_data = {
                ('code', 'code2func_edges', 'func'): code2func_edges,

                ('code', 'code2code_edges', 'code'): code2code_edges,
                ('code', 'codecall_edges', 'code'): codecall_edges,
                ('code', 'codexrefcode_edges', 'code'): codexrefcode_edges,
                ('code', 'codexrefdata_edges', 'data'): codexrefdata_edges,
                ('data', 'dataxrefcode_edges', 'code'): dataxrefcode_edges,
                ('data', 'dataxrefdata_edges', 'data'): dataxrefdata_edges,
            }
            num_nodes_dict = {'code': len(codenodes), 'data': len(datanodes), 'func': len(funcnodelookup)}
            g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

            #normlization
            new_min, new_max = 0, 1
            for o in codenodes:
                o.addr = th.tensor((o.addr - addr_min) / (
                        addr_max - addr_min) * (new_max - new_min) + new_min)
                o.func_addr = th.tensor((o.func_addr - addr_min) / (
                        addr_max - addr_min) * (new_max - new_min) + new_min)
                o.embeddings = th.tensor(numpy.array(list(o.embeddings)+[[0]*128]*(Ninst_addrs - len(o.embeddings)))).view(-1)
                o.embeddings = th.cat((o.addr.view(1), o.func_addr.view(1), o.embeddings)).float()
                o.avg = th.from_numpy(o.avg)
                o.avg = th.cat((o.addr.view(1), o.func_addr.view(1), o.avg)).float()

            for o in datanodes:
                o.addr = th.tensor((o.addr - addr_min) / (
                        addr_max - addr_min) * (new_max - new_min) + new_min)

            if len(codenodes) > 0:
                g.nodes['code'].data['feat'] = th.stack(([o.embeddings for o in codenodes]))
                g.nodes['code'].data['featmean'] = th.stack(([o.avg for o in codenodes]))
            if len(datanodes) > 0:
                g.nodes['data'].data['feat'] = th.stack(([o.addr.view(1) for o in datanodes]))

            savebin(filemd5=binaries[:-len(jsonex)], g=g,GT_edges=GT_edges,funcsave=funcsave)

print('total: %d\nskipped: %d\nprocessed: %d\nicalls: %d\nicallsite: %d\n'% (total, skipped, processed, icalls, icallsite))
timetotal = time.time()-starttime
print("timetotal: ", timetotal)