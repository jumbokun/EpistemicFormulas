import copy
import matplotlib.pyplot as plt
false_id = 0
true_id = 1

import random
import time
import signal
import json
import platform

SAVE_FOLDER = "result"
# Table T(u)-> (i,l,h) and H(i,l,h)-> u
# T = {}
# H = {}
# string: Node

MAXSIZE=10000000000
def handler(signum, frame):
    raise TimeoutError

# Only set up signal handler on Unix-like systems
if platform.system() != 'Windows':
    signal.signal(signal.SIGALRM, handler)
kp_threshold = 100000

node_counter = 2
branch_cache = {}
conjoin_cache = {} 
negate_cache = {}
symbol_2_number = {}
number_2_symbol = {}
nodeID_2_key = {}
K_flat_cache = {}
kp_sat_cache = {}
conjoin_sat_cache={}
good_order = 0
bad_order = 0
kp_sat_cache_usage = 0
conjoin_sat_cahce_usage = 0


def reset_cache():
    global node_counter
    global branch_cache
    global conjoin_cache
    global negate_cache
    global symbol_2_number
    global number_2_symbol
    global nodeID_2_key
    global false_node
    global true_node
    global K_flat_cache
    global good_order
    global bad_order
    global kp_sat_cache
    global kp_sat_cache_usage
    global conjoin_sat_cache
    global conjoin_sat_cache_usage

    good_order = 0
    bad_order = 0
    kp_sat_cache_usage = 0
    conjoin_sat_cahce_usage=0
    node_counter = 2
    branch_cache = {}
    conjoin_cache = {} 
    negate_cache = {}
    symbol_2_number = {}
    number_2_symbol = {}
    nodeID_2_key = {}   
    K_flat_cache = {}
    kp_sat_cache={}
    conjoin_sat_cahce={}
    Node.delete_all_instances()
    false_node = Node(id=0, var_id=None, when0=None, when1=None)
    true_node = Node(id=1, var_id=None, when0=None, when1=None)
    branch_cache["false"] = false_node
    branch_cache["true"] = true_node
    nodeID_2_key[false_id]="false"
    nodeID_2_key[true_id]="true"

# class Variable:
#     Bool_cache = {0: "F", 1: "T"}
#     Epic_cache = {}
#     var_list = {}
#     def __init__(self):
#         pass

class Node:
    _instances = []
    def __init__(self, id, var_id, when1, when0, is_enode=False):
        self.id = id
        self.var_id = var_id
        self.when0 = when0
        self.when1 = when1
        self.is_enode = is_enode


        Node._instances.append(self)
        # self.max_id = self.get_max_id() # return maximal node id for objective sub-formula
        # return

    # def get_max_id(self):
    #     if self.id == true_id or self.id == false_id:
    #         return 0
    #     if self.var_id > 0:
    #         return max(self.id, self.when0.max_id, self.when1.max_id)
    #     else:
    #         return max(-self.var_id, self.when0.max_id, self.when1.max_id)
        
    @classmethod
    def delete_all_instances(cls):
        cls._instances.clear()

def get_key(head_id, when0: Node, when1: Node)-> str:
    if head_id < 0:
        key = "K(" + str(-head_id) + ")?" + str(when1.id) + ":" + str(when0.id)
    else:
        key = str(head_id) + "?" + str(when1.id) + ":" + str(when0.id)
    return key

def mk_branch(var_id, when0: Node, when1: Node)-> Node:
    if when0.id == when1.id:
        return when0
    # key = str(var_id) + ":" + str(when1.id) + ":" + str(when0.id) # var_id : when1.id : when0.id
    key = get_key(head_id=var_id, when0=when0, when1=when1)
    if key not in branch_cache.keys():
        counter = len(branch_cache)
        node = Node(id=counter, var_id=var_id, when1= when1, when0 = when0, is_enode=False)
        branch_cache[key] = node
        nodeID_2_key[counter] = key
        # print("create new node")
        # print(branch_cache.keys())
        return node
    else:
        return branch_cache[key]


def mk_know(node_id, when0: Node, when1: Node)-> Node:
    knode = branch_cache[nodeID_2_key[node_id]]
    assert knode.var_id >= 0
    
    key = get_key(head_id=-node_id, when0=when0, when1=when1)

    if key not in branch_cache.keys():
        counter = len(branch_cache)
        enode = Node(id=counter, var_id=-node_id, when1= when1, when0 = when0, is_enode=True) # variable ID defined as the negation of node ID
        branch_cache[key] = enode
        nodeID_2_key[counter] = key
        # print("create new node")
        # print(branch_cache.keys())
        return enode
    else:
        return branch_cache[key]
    
def negate(node: Node) -> Node:
    if node.id == true_id:
        return false_node
    if node.id == false_id:
        return true_node
    if node.id not in negate_cache.keys():
        # print("node.id = " + str(node.id))
        rlt = mk_branch(var_id=node.var_id, when0=negate(node.when0), when1=negate(node.when1))
        # if rlt.size > kp_threshold:
        #     # print("negate sizebeyond threshold")
        #     ori_size = rlt.size
        #     # print("Size before KP: {}, after KP: {}\n################".format(str(ori_size),str(rlt.size)))
        negate_cache[node.id] = rlt
        # print(negate_cache)
        return rlt
    else:
        # print("node.id = " + str(node.id))
        return negate_cache[node.id]

    

def conjoin(lhs: Node, rhs : Node) -> Node:
    # print(branch_cache)
    # print(conjoin_cache)
    # print(negate_cache)
    # print(symbol_2_number)
    if lhs.id == rhs.id:
        return lhs
    if lhs.id == false_id or rhs.id == false_id:
        return branch_cache["false"]
    if lhs.id == true_id:
        return rhs
    if rhs.id == true_id:
        return lhs
    if (lhs.var_id > rhs.var_id):
        tmp = lhs
        lhs = rhs
        rhs = tmp
    key = str(lhs.id) + ":" + str(rhs.id)
    # if (lhs.var_id > rhs.var_id):
    #     key = str(rhs.id) + ":" + str(lhs.id)
    # else:
    #     key = str(lhs.id) + ":" + str(rhs.id)
    if key not in conjoin_cache.keys():
        # compare variables
        if lhs.var_id >0:
            if (lhs.var_id == rhs.var_id):
                rlt= mk_branch(var_id=lhs.var_id, when0=conjoin(lhs=lhs.when0, rhs=rhs.when0), when1=conjoin(lhs=lhs.when1, rhs= rhs.when1))
            else: # (lhs.var_id < rhs.var_id):
                rlt= mk_branch(var_id=lhs.var_id, when0=conjoin(lhs=lhs.when0, rhs=rhs), when1=conjoin(lhs=lhs.when1, rhs= rhs))
            # else: # (lhs.var_id > rhs.var_id):
            #     rlt= mk_branch(var_id=rhs.var_id, when0=conjoin(rhs=lhs, lhs=rhs.when0), when1=conjoin(rhs=lhs, lhs= rhs.when1))
        else:
            if (lhs.var_id == rhs.var_id):
                rlt= mk_know(node_id=-lhs.var_id, when0=conjoin(lhs=lhs.when0, rhs=rhs.when0), when1=conjoin(lhs=lhs.when1, rhs= rhs.when1))
            else: #(lhs.var_id < rhs.var_id):
                # print(-lhs.var_id)
                # print(nodeID_2_key)
                rlt= mk_know(node_id=-lhs.var_id, when0=conjoin(lhs=lhs.when0, rhs=rhs), when1=conjoin(lhs=lhs.when1, rhs= rhs))
            # else: # (lhs.var_id > rhs.var_id):
            #     rlt= mk_know(var_id=rhs.var_id, when0=conjoin(rhs=lhs, lhs=rhs.when0), when1=conjoin(rhs=lhs, lhs= rhs.when1))
        # if rlt.size > kp_threshold:
        #     # print("conjoin size beyond threshold")
        #     ori_size = rlt.size
        #     # print("Size before KP: {}, after KP: {}\n################".format(str(ori_size),str(rlt.size)))
        conjoin_cache[key] = rlt

    return conjoin_cache[key]

def disjoin(lhs: Node, rhs: Node) -> Node:
    if lhs.id == true_id or rhs.id == true_id:
        return true_node
    if lhs.id == false_id:
        return rhs
    if rhs.id == false_id:
        return lhs
    if rhs.id == lhs.id:
        return rhs
    return negate(conjoin(lhs = negate(lhs), rhs=negate(rhs)))

def implies(premise: Node, conclusion: Node) -> Node:
    if premise.id == true_id:
        return conclusion
    if premise.id == false_id:
        return true_node
    if conclusion.id == true_id:
        return true_node
    if conclusion.id == false_id:
        return negate(premise)
    if premise.id == conclusion.id:
        return true_node
    return disjoin(negate(premise), conclusion)


def display(x: Node)-> str:
    if x.id == true_id: 
        return "true"
    if x.id == false_id:
        return "false"
    if x.var_id > 0:
        rlt = "(" + str(number_2_symbol[x.var_id]) + "? " + display(x.when1) + ": " + display(x.when0) + ")"
    else:
        rlt = "(K" + str(display(branch_cache[nodeID_2_key[-x.var_id]])) + "? " + display(x.when1) + ": " + display(x.when0) + ")"

    return rlt


# function NOT(node: ENode): ENode { return negate(node); }
# function OR(...args: ENode[]) { var result = _f; for(var i = 0; i < arguments.length; i++) result = disjoin(result, arguments[i]); return result; }; 
# function AND(...args: ENode[]) { var result = _t; for(var i = 0; i < arguments.length; i++) result = conjoin(result, arguments[i]); return result; };


def V(x: str)-> Node:
    if x in symbol_2_number:
        return mk_branch(var_id= symbol_2_number[x], when0= false_node, when1= true_node)
    index = len(symbol_2_number)+1
    number_2_symbol[index] = x
    symbol_2_number[x] = index
    node = mk_branch(var_id= index, when0= false_node, when1= true_node)
    return node

def denesting(var_id, when0:Node, when1:Node)-> Node:
    global good_order
    global bad_order
    assert var_id < 0
    if when0.var_id==None:
        var_when0 = 0
    else: 
        var_when0 = when0.var_id

    if when1.var_id==None:
        var_when1 = 0
    else: 
        var_when1 = when1.var_id
    k0 = K(when0)
    k1 = K(when1)
    if (k0.var_id== None or var_id <= k0.var_id) and (k1.var_id== None or var_id<= k1.var_id):
        good_order += 1
        return mk_know(node_id=-var_id, when0=k0, when1=k1)
    else:
        bad_order +=1
        khead = K(branch_cache[nodeID_2_key[-var_id]])
        return conjoin(disjoin(khead, k0), disjoin(k1, NOT(khead)))

def K(node: Node) -> Node:
    global K_flat_cache
    key = node.id
    result = K_flat_cache.get(key)
    if result == None:
        if node.id == true_id:
            result = true_node
        elif node.id == false_id:
            result = false_node
        # assert node.var_id >=0  # node test an epistemic variable, should be an error
        elif node.var_id < 0: # avoid nesting of Knowledge
            # print("knode:" + display(node))
            # print("when1: " + display(node.when1))
            # print("when0: "+ display(node.when0))
            # print("first disjoin: " + display(disjoin(knode, K(when0))))
            # print("First conjoin: " + display(conjoin(disjoin(knode, K(when0)), disjoin(K(when1), NOT(knode)))))
            khead = K(branch_cache[nodeID_2_key[-node.var_id]])
            # print("khead: " + display(khead))
            # print("Nesting solving: " + display(conjoin(conjoin(disjoin(khead, K(node.when0)), disjoin(K(node.when1), NOT(khead))), K(disjoin(node.when0, node.when1)))))

            # 09.04: Test alternative approach to handle nesting K
            # result = conjoin(conjoin(disjoin(khead, K(node.when0)), disjoin(K(node.when1), NOT(khead))), K(disjoin(node.when0, node.when1)))
            # result = mk_know(node_id= -node.var_id, when0=K(node.when0), when1=K(node.when1))
            # result = reorder(head= khead, when0=K(node.when0), when1=K(node.when1))
            result = denesting(var_id=node.var_id, when0=node.when0, when1=node.when1)

        else:
            result = mk_know(node_id = node.id, when0=false_node, when1=true_node)
        ######
        # Apr. 10
        # if result.size > kp_threshold:
        #     # print("K size beyond threshold")
        #     ori_size = result.size
            # print("Size before KP: {}, after KP: {}\n################".format(str(ori_size),str(result.size)))
        ###
        K_flat_cache[key] = result
    return result

# def substitute(node: Node, evar: int, branch_1: bool) -> Node:
#     if node.id== true_id or node.id == false_id:
#         return node
#     if node.var_id > 0:
#         return node
#     assert evar < 0
#     if node.var_id == evar:
#         if branch_1:
#             return substitute(node.when1, evar=evar, branch_1=branch_1)
#         else:
#             return substitute(node.when0, evar=evar, branch_1=branch_1)
#     else:
#         new_when0 = substitute(node.when0, evar=evar, branch_1=branch_1)
#         new_when1 = substitute(node.when1, evar=evar, branch_1=branch_1)
#         return mk_know(node_id=-node.var_id, when0=new_when0, when1=new_when1)

def rt_evar_list(node:Node):
    if node.id == true_id or node.id == false_id:
        return []
    if node.var_id >= 0:
        return []
    return list(set(rt_evar_list(node.when0)+rt_evar_list(node.when1)+[node.var_id]))

def rt_nodes_list(node:Node):
    if node.id == true_id or node.id == false_id:
        return [node.id]
    else:
        return list(set(rt_nodes_list(node=node.when0)+rt_nodes_list(node=node.when1)+[node.id]))


def rt_edges_list(node:Node):
    if node.id == true_id or node.id == false_id:
        return []
    else:
        edge1 = "{}|1|{}".format(str(node.id), str(node.when1.id))
        edge0 = "{}|0|{}".format(str(node.id), str(node.when0.id))
        return list(set([edge0]+[edge1] + rt_edges_list(node.when0) + rt_edges_list(node.when1)))

def rt_Edep(node:Node):
    if node.id == true_id or node.id == false_id:
        return []
    if node.var_id>=0:
        return []
    else:
        if len(rt_Edep(node.when0))>len(rt_Edep(node.when1)):
            rlt = rt_Edep(node.when0)
            rlt.append(-node.var_id)
        else:
            rlt = rt_Edep(node.when1)
            rlt.append(-node.var_id)
        return rlt
    
def rt_pos_Edep(node:Node):
    if node.id == true_id or node.id == false_id:
        return []
    if node.var_id>=0:
        return []
    else:
        if len(rt_pos_Edep(node.when0))>len(rt_pos_Edep(node.when1)):
            rlt = rt_pos_Edep(node.when0)
        else:
            rlt = rt_pos_Edep(node.when1)
            rlt.append(-node.var_id)
        return rlt

def concise_check(node:Node, formula:Node):
    if node.id == true_id or node.id == false_id:
        return True
    if node.var_id>=0:
        return True
    knode = branch_cache[nodeID_2_key[-node.var_id]]
    pos_rlt = conjoin(formula, K(knode))
    neg_rlt = conjoin(formula, negate(K(knode)))
    if pos_rlt.id == false_id:
        return False
    if neg_rlt.id == false_id:
        return False
    return concise_check(node.when1, pos_rlt) and concise_check(node.when0, neg_rlt)
# def reordering(node: Node)-> Node:
#     evar_list = rt_evar_list(node=node)
#     evar_list.sort(reverse=True)
#     # print(evar_list)
#     print("initial formula: " + display(node))
#     print("# e-variables:{}".format(str(len(evar_list))))
#     print("initial size:{}".format(str(len(rt_nodes_list(node)))))
#     rlt = node
#     for evar in evar_list:
#         rlt = mk_know(node_id=-evar, when0= substitute(node=rlt,evar=evar,branch_1=False),when1= substitute(node=rlt,evar=evar,branch_1=True))
#         print("\treorder var {}, diagram size {}, cache size {}".format(str(evar), str(len(rt_nodes_list(rlt))), str(len(branch_cache)+ len(K_flat_cache))))
#     print("reordered size:{}".format(str(len(rt_nodes_list(rlt)))))
#     print("#################\n")
#     return rlt



def NOT(node : Node)-> Node:
    return negate(node)

def OR(node1: Node, node2: Node) -> Node:
    return disjoin(node1, node2)

def AND(node1: Node, node2: Node) -> Node:
    return conjoin(node1, node2)

def conjoin_sat(lhs:Node, rhs:Node):
    global conjoin_sat_cache
    global conjoin_sat_cahce_usage
    if lhs.id == false_id and rhs.id == false_id:
        return False
    if lhs.id == true_id or rhs.id == true_id:
        return True
    if lhs.id == false_id:
        return conjoin_sat(rhs.when0, rhs.when1) 
    if rhs.id == false_id:
        return conjoin_sat(lhs.when0, lhs.when1) 
    if (lhs.var_id > rhs.var_id):
        tmp = lhs
        lhs = rhs
        rhs = tmp
    # if (lhs.var_id > rhs.var_id):
    #     key = str(rhs.id) + ":" + str(lhs.id)
    # else:
    #     key = str(lhs.id) + ":" + str(rhs.id)
    key = str(lhs.id) + ":" + str(rhs.id)
    rlt = conjoin_sat_cache.get(key)
    if rlt == None:
        if (lhs.var_id == rhs.var_id):
            if conjoin_sat(lhs.when0, rhs.when0):
                rlt=True
            elif conjoin_sat(lhs.when1, rhs.when1):
                rlt=True
            else:
                rlt = False
        else: # (lhs.var_id < rhs.var_id):
            if conjoin_sat(lhs.when0, rhs):
                rlt = True
            elif conjoin_sat(lhs.when1, rhs):
                rlt = True
            else:
                rlt = False
        conjoin_sat_cache[key]=rlt
    else:
        conjoin_sat_cahce_usage +=1
    return rlt
    # else: # (lhs.var_id > rhs.var_id):
    #     rlt= mk_branch(var_id=rhs.var_id, when0=conjoin(rhs=lhs, lhs=rhs.when0), when1=conjoin(rhs=lhs, lhs= rhs.when1))


def is_sat(klist, nlist)-> bool:
    global kp_sat_cache
    global kp_sat_cache_usage
    assert len(klist)==1 and len(nlist)>=1
    kid = klist[0]
    know = branch_cache[nodeID_2_key[kid]]
    assert know.var_id == None or know.var_id>=0
    # is_sat = True
    for nid in nlist:
        sat_key = "{}|{}".format(str(kid),str(nid))
        rlt_node = kp_sat_cache.get(sat_key)
        if rlt_node == None:
            neg_know = branch_cache[nodeID_2_key[nid]]
            assert neg_know.var_id == None or neg_know.var_id>=0
            rlt_node = conjoin(lhs=know, rhs=negate(branch_cache[nodeID_2_key[nid]]))
            kp_sat_cache[sat_key]=rlt_node
        else:
            kp_sat_cache_usage +=1
        if rlt_node.id == false_id:
            return False
    return True
    

def propagate_Knowledge(node: Node, klist=[true_id], nlist=[false_id], use_s5= False)-> Node:

    if node.id == true_id or node.id == false_id:
        return node
    when0 = node.when0
    when1 = node.when1
    if node.var_id > 0:
        # when0_nlist = nlist
        # when1_klist = klist
        if use_s5:
            # print("s5 return: " + display(disjoin(node, negate(branch_cache[nodeID_2_key[klist[0]]]))))
            premise = branch_cache[nodeID_2_key[klist[0]]]
            prop = V(number_2_symbol[node.var_id])
            if implies(premise=premise, conclusion=prop).id==true_id:
                
                propagate_node = propagate_Knowledge(node=node.when1, use_s5=use_s5, klist=klist, nlist=nlist)
            elif implies(premise=premise, conclusion=negate(prop)).id==true_id:
                propagate_node = propagate_Knowledge(node=node.when0, use_s5=use_s5, klist=klist, nlist=nlist)
            else:
                when1_klist = [conjoin(lhs=premise, rhs=prop).id]
                when0_klist = [conjoin(lhs=premise, rhs=negate(prop)).id]
                propagate_node = mk_branch(var_id=node.var_id, when1=propagate_Knowledge(node=node.when1, use_s5=use_s5, klist=when1_klist, nlist=nlist),
                                when0=propagate_Knowledge(node=node.when0, use_s5=use_s5, klist=when0_klist, nlist=nlist))
        else:
            propagate_node = node
    else:
        when0_klist = klist
        when1_nlist = nlist
        when1_klist = [conjoin(lhs=branch_cache[nodeID_2_key[klist[0]]], rhs=branch_cache[nodeID_2_key[-node.var_id]]).id]
        when0_nlist = sorted(set(nlist + [-node.var_id]))
        is_sat1 = is_sat(when1_klist, when1_nlist)
        is_sat0 = is_sat(when0_klist, when0_nlist)
        assert is_sat1 or is_sat0
        if not is_sat1:
            propagate_node= propagate_Knowledge(node=when0, klist=when0_klist, nlist=when0_nlist, use_s5=use_s5)
        elif not is_sat0:
            propagate_node= propagate_Knowledge(node=when1, klist=when1_klist, nlist=when1_nlist, use_s5=use_s5)
        
        # if use_s5 and node.var_id< 0:
        #     new_when1 = propagate_Knowledge(disjoin(when1, negate(branch_cache[nodeID_2_key[-node.var_id]])), klist=when1_klist, nlist=when1_nlist) # TODO: proof
        #     # new_when1 = propagate_Knowledge(conjoin(when1, branch_cache[nodeID_2_key[-node.var_id]]), klist=when1_klist, nlist=when1_nlist) # TODO: proof

        else:
            new_when1 = propagate_Knowledge(when1, klist=when1_klist, nlist=when1_nlist,use_s5=use_s5)
            new_when0 = propagate_Knowledge(when0, klist=when0_klist, nlist=when0_nlist, use_s5=use_s5)
            if new_when0.id == new_when1.id:
                propagate_node = new_when0
            elif node.var_id>0:
                propagate_node = mk_branch(var_id=node.var_id, when0=new_when0, when1 =new_when1)
            else:
                propagate_node = mk_know(node_id=-node.var_id, when0=new_when0, when1=new_when1)
        # if 2*len(rt_edges_list(node)) < len(rt_edges_list(propagate_node)):
        #     print("original size:{}, Kp size:{}".format(str(len(rt_nodes_list(node))), str(len(rt_nodes_list(propagate_node)))))
        #     # print(rt_nodes_list(formula))
        #     print("########################")
    return propagate_node


# phi = implies(OR(V("p"), V("q")), V("p"))
# phi = implies(AND(V("p"),V("q")), OR(V("q"), V("p")))
# phi = AND(K(V("p")), K(OR(V("p"),V("q"))))
# phi = implies(AND(K(OR(V("p"),V("q"))),K(V("p"))), OR(K(OR(V("q"), NOT(V("r")))), V("r")))
# phi = AND(K(V("p")), NOT(V("p")))
# phi = implies(K(V("p")), NOT(K(NOT(V("p")))))
# phi = OR(V("p"), implies(V("q"), NOT(K(V("r")))))

# vp = V("p")
# vq = V("q")
# vr = V("r")
# phi = AND(AND(K(OR(vp, vq)),NOT(K(vp))), NOT(K(AND(vp, vq))))
# phi = AND(K(OR(vp, vq)), NOT(K(vp)))

# print(display(phi))
# print(display(propagate_Knowledge(phi)))
# print(branch_cache)

# phi_propagate = propagate_Knowledge(phi)
# phi = OR(V("p"), V("q"))
# print(phi.id)
# print(phi.when0.id)
# print(phi.when1.id)
# print(branch_cache) 

def gimea_formula(num_var, complexity, deg_nesting=1):
    if complexity ==1:
        var_dice = random.randint(0,num_var-1)
        return V('v'+str(var_dice))
    if complexity >=3:
        if deg_nesting==0:
            con_dice = random.randint(0,2)
        else:
            con_dice = random.randint(0,3)
    else:
        if deg_nesting==0:
            con_dice = 0
        else:
            con_dice = random.choice([0,3])
    """
        connective dice =
        0: negation
        1: conjunction
        2: disjunction
        3: K modality
    """
    if con_dice==0:
        return negate(gimea_formula(num_var=num_var, complexity=complexity-1, deg_nesting=deg_nesting))
    elif con_dice==1:
        complex_dice = random.randint(1,complexity-2)
        return AND(gimea_formula(num_var=num_var, complexity=complex_dice, deg_nesting=deg_nesting), \
                   gimea_formula(num_var=num_var, complexity=complexity-complex_dice, deg_nesting=deg_nesting))
    elif con_dice==2:
        complex_dice = random.randint(1,complexity-2)
        return OR(gimea_formula(num_var=num_var, complexity=complex_dice, deg_nesting=deg_nesting), \
                  gimea_formula(num_var=num_var, complexity=complexity-complex_dice, deg_nesting=deg_nesting))
    else:
        return K(gimea_formula(num_var=num_var, complexity=complexity-1, deg_nesting=deg_nesting-1))

def dep(node:Node):
    if node.id==true_id or node.id==false_id:
        return 1
    return max(dep(node.when0), dep(node.when1))+1


def test_ordering(node:Node):
    if node.id == true_id or node.id == false_id:
        return
    if node.when0.var_id is not None:
        assert node.var_id < node.when0.var_id
        if not (node.when0.id==true_id or node.when0.id == false_id):
            test_ordering(node=node.when0)
    if node.when1.var_id is not None:
        assert node.var_id < node.when1.var_id
        if not (node.when1.id==true_id or node.when1.id == false_id):
            test_ordering(node=node.when1)

def test_out(name, axiom, sat_check=False, use_s5=False)-> bool:
    print("--- " + name )
    print("     initially: " + display(axiom))
    print(branch_cache.keys())
    result = propagate_Knowledge(axiom, use_s5=use_s5)
    print("    propagated: " + display(result))
    if not sat_check:
        if result.id != true_id:
            print("    conclusion: axiom rejected") 
            return False
        else:
            print("    conclusion: axiom accepted")
            return True
    else:
        if result.id != false_id:
            print("    conclusion: formula satisfiable")
            return True
        else:
            print("    conclusion: formula unsatisfiable")
            return False

#####################
# axioms test cases

# false_node = Node(id=0, var_id=None, when0=None, when1=None)
# true_node = Node(id=1, var_id=None, when0=None, when1=None)
# branch_cache["false"] = false_node
# branch_cache["true"] = true_node
# nodeID_2_key[false_id]="false"
# nodeID_2_key[true_id]="true"
reset_cache()
# test_out("reflexivity", implies(K(V("p")), V("p")), use_s5=True)
# test_out("reflexivity2", implies(K(AND(V("p"), NOT(V("r")))), V("p")), use_s5=True)
# test_out("truthfulness", AND(K(V("p")),AND(NOT(V("p")), V("q"))), use_s5=True, sat_check=True)
test_out("S5", implies(K(implies(V("p"), V("q"))),implies(negate(V("q")), negate(K(V('p'))))), use_s5=True)
# test_out("knowing true", K(true_node))
# test_out("knowing true 2", K(implies(V("p"), OR(V("p"), V("q")))))
# test_out("consistency", implies(K(V("p")), NOT(K(NOT(V("p"))))))
# test_out("false belief", AND(K(V("p")), NOT(V("p"))), sat_check=True)
# test_out("conjunction distribution 1", implies(conjoin(K(V("p")), K(V("q"))), K(conjoin(V("p"), V("q")))))
# test_out("conjunction distribution 2", implies(K(conjoin(V("p"), V("q"))), conjoin(K(V("p")), K(V("q")))))
# test_out("modus ponendo ponens", implies(conjoin(K(implies(V("p"), V("q"))), K(V("p"))), K(V("q"))))
# test_out("positive introspection", implies(K(V("p")), K(K(V("p")))))
# test_out("positive introspection 2", implies(AND(K(V("q")),K(V("p"))), K(AND(V("q"),K(V("p")))))) # (Kq \wedge Kp)->(K(q \wedge K(p))
# test_out("positive introspection 3", implies(AND(K(V("q")),K(V("p"))), K(AND(K(V("q")),K(V("p")))))) # (Kq \wedge Kp)->K 

# test_out("negative Introspection", implies(NOT(K(V("p"))), K(NOT(K(V("p"))))))
# test_out("negative Introspection 2", implies(NOT(K(AND(V("q"),K(V("p"))))), K(NOT(K(AND(V("q"),K(V("p"))))))))


############################


num_test = 200
num_var=30
max_len = 501
len_gap = 25
dropout=0.1
# max_nesting = 5
# for num_var in range (5,30,5):
# var_list = [15,30,45]
max_nesting=0

avg_kp_size_dict = {}
avg_og_size_dict = {}
# worst_kp_size_list = []
time_dict = {}
com_list = range(len_gap,max_len,len_gap)

colors = ['red', 'green', 'blue','black','cyan','magenta']


        
for c in com_list:
    for deg_nesting in [0,1]:
        for num_var in [15,30,45]:
            exp_key = "({},{})".format(str(deg_nesting),str(num_var))
            if exp_key not in avg_kp_size_dict:
                avg_kp_size_dict[exp_key]=[]
                avg_og_size_dict[exp_key]=[]
                time_dict[exp_key]=[]
            
            start_time = time.time()
            c_og_size_list = []
            c_avg_size_list = []
            c_time_list = []
            count_maxsize=0
            # ori_sum = 0

            for i in range(num_test):
                cur_start = time.time()

                complexity = c + random.randint(0,len_gap)
                # print("num_var: " + str(num_var) + ", complexity: " + str(complexity))

                ####################
                # reset caches
                reset_cache()
                #######################
                # reset_cache()
                # print(len(branch_cache))
                assert len(branch_cache)==2
                try:
                    max_duration = 10
                    # Only use signal.alarm on Unix-like systems
                    if platform.system() != 'Windows':
                        signal.alarm(max_duration)
                    formula = gimea_formula(num_var=num_var, complexity=complexity, deg_nesting=deg_nesting)

                # print(display(formula))
                    kp_form = propagate_Knowledge(formula)
                    # kp_form = formula
                    # if 2*len(rt_edges_list(formula)) < len(rt_edges_list(kp_form)):
                    #     print("original size:{}, Kp size:{}".format(str(len(rt_edges_list(formula))), str(len(rt_edges_list(kp_form)))))
                    #     # print(display(formula))
                    #     print("########################")
                    #     # print(display(kp_form))
                    # if 2*len(rt_nodes_list(formula)) < len(rt_edges_list(kp_form)):
                    #     print("original size:{}, Kp size:{}".format(str(len(rt_nodes_list(formula))), str(len(rt_nodes_list(kp_form)))))
                    #     # print(rt_nodes_list(formula))
                    #     print("########################")
                    #     # print(rt_nodes_list(kp_form))
                    #     assert False
                    # assert len(rt_nodes_list(formula)) >= len(rt_nodes_list(kp_form))
                    if platform.system() != 'Windows':
                        signal.alarm(0)
                except TimeoutError:
                    print("Test{}: duration more than {}s".format(str(i), str(max_duration)))
                    kp_size = MAXSIZE
                    c_avg_size_list.append(kp_size)
                    duration = MAXSIZE
                    c_time_list.append(duration)
                    count_maxsize +=1
                    assert count_maxsize < dropout*num_test
                    continue
                # test_ordering(kp_form)
                # print(display(kp_form))
                # print("formula_depth: " + str(dep(formula)) + " kp_depth: " + str(dep(kp_form)))
                # print("##########")
                # ori_sum += len(rt_edges_list(formula))
                cur_end = time.time()
                duration = cur_end - cur_start
                c_time_list.append(duration)
                # print("Test{}".format(str(i)))
                evar_size = len(rt_evar_list(kp_form))
                og_evar_size = len(rt_evar_list(formula))
                og_size = len(rt_nodes_list(formula))
                kp_size = len(rt_nodes_list(kp_form))
                # if len(rt_pos_Edep(kp_form)) > num_var:
                #     # check_rlt = concise_check(node=kp_form, formula=true_node)
                #     print("num_var:{}, Edep:{}, pos_Edep:{}".format(str(num_var), str(len(rt_Edep(kp_form))),str(len(rt_pos_Edep(kp_form)))))
                #     if len(rt_Edep(kp_form)) >= 2**(num_var+1):
                #         print(display(kp_form))
                #         assert False 
                    # assert concise_check(node=kp_form, formula=true_node)
                    # print(rt_Edep(kp_form))
                # assert rt_Edep(kp_form)<= num_var
                c_og_size_list.append(og_size)
                c_avg_size_list.append(kp_size)
                
                if duration > 0.1:
                    pass
                    print("Test {}-{}: duration:{}, #Evar: {}->{}, Size: {}->{}".format(exp_key,str(i),str(duration), str(og_evar_size),str(evar_size),str(og_size),str(kp_size)))
                # if good_order + bad_order >0:
                #     print("good_order={}, bad_order={}, good ratio {}%".format(str(good_order), str(bad_order), str(100.0*good_order/(good_order+bad_order))))
            end_time = time.time()
            c_avg_size_list.sort()
            c_og_size_list.sort()
            c_time_list.sort()
            list_len = len(c_avg_size_list)
            assert list_len == len(c_time_list)
            size_avg = sum(c_avg_size_list[int(list_len*dropout):list_len-int(list_len*dropout)])*1.0/len(c_avg_size_list[int(list_len*dropout):list_len-int(list_len*dropout)])
            og_size_avg = sum(c_og_size_list[int(list_len*dropout):list_len-int(list_len*dropout)])*1.0/len(c_og_size_list[int(list_len*dropout):list_len-int(list_len*dropout)])
            time_avg = sum(c_time_list[int(list_len*dropout):list_len-int(list_len*dropout)])*1.0/len(c_time_list[int(list_len*dropout):list_len-int(list_len*dropout)])
            avg_kp_size_dict[exp_key].append(size_avg)
            avg_og_size_dict[exp_key].append(og_size_avg)
            time_dict[exp_key].append(time_avg*1000)
            print("Finish running examples of length:{}, using time {} s, 5-95% {}".format(str(c), str(end_time - start_time), str(time_avg*num_test*(1-2*dropout))))
            ###
            # axiom test
            # reset_cache()
            # complexity = c + random.randint(0,len_gap)
            # alpha = gimea_formula(num_var=num_var, complexity=complexity, deg_nesting=max_nesting)
            # complexity = c + random.randint(0,10)
            # beta = gimea_formula(num_var=num_var, complexity=complexity, deg_nesting=max_nesting)

            # assert test_out("reflexivity", implies(K(alpha), alpha), use_s5=True)
            # assert not test_out("truthfulness", AND(K(alpha),AND(NOT(alpha), beta)), use_s5=True, sat_check=True)
            # assert test_out("knowing true 2", K(implies(alpha, OR(alpha, beta))))
            # assert test_out("consistency", implies(K(alpha), NOT(K(NOT(alpha)))))
            # # assert test_out("false belief", AND(K(alpha), NOT(alpha)), sat_check=True) or alpha.id == false_id or alpha.id == true_id
            # assert test_out("conjunction distribution 1", implies(conjoin(K(alpha), K(beta)), K(conjoin(alpha, beta))))
            # assert test_out("conjunction distribution 2", implies(K(conjoin(alpha, beta)), conjoin(K(alpha), K(beta))))
            # assert test_out("modus ponendo ponens", implies(conjoin(K(implies(alpha, beta)), K(alpha)), K(beta)))
            # assert test_out("positive introspection", implies(K(alpha), K(K(alpha))))
            # assert test_out("negative Introspection", implies(NOT(K(alpha)), K(NOT(K(alpha)))))
            #####
    with open("./{}/avg_og_size.json".format(SAVE_FOLDER), "w") as f:
        json.dump(avg_og_size_dict, f, indent=4)
    with open("./{}/avg_kp_size.json".format(SAVE_FOLDER), "w") as f:
        json.dump(avg_kp_size_dict, f, indent=4)
    with open("./{}/avg_time.json".format(SAVE_FOLDER), "w") as f:
        json.dump(time_dict, f, indent=4)

    fig, ax1 = plt.subplots()
    for deg_nesting in [0,1]:
        
            # exp_key = "({},{})".format(str(deg_nesting),str(num_var))
        output_length = len(avg_kp_size_dict["({},30)".format(str(deg_nesting))])
        for num_var in [15,30,45]:
            ax1.plot(com_list[:output_length], avg_kp_size_dict["({},{})".format(str(deg_nesting),str(num_var))], 
                     label="dep={},n={}".format(str(deg_nesting),str(num_var)), color=colors[int(deg_nesting*3+ num_var/15-1)])
    # plt.plot(com_list, avg_ori_size_list, label="avg_ori", color="green")
    # ax1.plot(com_list, worst_kp_size_list, label="worst size", color="blue")
    ax1.set_xlabel("Sentence length")
    ax1.set_ylabel("#Nodes")
    ax1.legend(loc="upper left")
    # plt.title("Average #Edge")
    plt.savefig("./{}/dep0-1_length{}_size.png".format(SAVE_FOLDER,str(max_len)))
    plt.yscale("log")
    plt.savefig("./{}/dep0-1_length{}_size_logscale.png".format(SAVE_FOLDER,str(max_len)))
    plt.clf()
    plt.close()

    fig, ax1 = plt.subplots()
    for deg_nesting in [0,1]:
        
            # exp_key = "({},{})".format(str(deg_nesting),str(num_var))
        output_length = len(time_dict["({},30)".format(str(deg_nesting))])
        for num_var in [15,30,45]:
            ax1.plot(com_list[:output_length], time_dict["({},{})".format(str(deg_nesting),str(num_var))], 
                     label="dep={},n={}".format(str(deg_nesting),str(num_var)), color=colors[int(deg_nesting*3+ num_var/15-1)])
    # plt.plot(com_list, avg_ori_size_list, label="avg_ori", color="green")
    # ax1.plot(com_list, worst_kp_size_list, label="worst size", color="blue")
    ax1.set_xlabel("Sentence length")
    ax1.set_ylabel("Time(ms)")
    ax1.legend(loc="upper left")
    # plt.title("Average Time")
    plt.savefig("./{}/dep0-1_length{}_time.png".format(SAVE_FOLDER,str(max_len)))
    plt.yscale("log")
    plt.savefig("./{}/dep0-1_length{}_time_logscale.png".format(SAVE_FOLDER,str(max_len)))
    plt.clf()
    plt.close()


# print(com_list)
# print(time_dict)
# print(avg_kp_size_dict)
    # plt.cla()


        