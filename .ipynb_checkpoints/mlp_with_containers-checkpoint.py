import gurobipy as gp
from gurobipy import GRB
import re
import argparse

# ------------------------------------------
# 1. Read in the data
# ------------------------------------------
def parse_mcgb_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    content = re.sub(r'#.*', '', content)
    statements = [s.strip() for s in content.split(';') if s.strip()]

    data = {}
    for stmt in statements:
        if stmt.startswith('set Nodos:='):
            data['N'] = [int(x) for x in stmt.replace('set Nodos:=', '').split()]
            data['Depot'] = data['N'][0]
        elif stmt.startswith('set I:='):
            data['N_farms'] = [int(x) for x in stmt.replace('set I:=', '').split()]
        elif stmt.startswith('set K:='):
            data['K'] = [int(x) for x in stmt.replace('set K:=', '').split()]
        elif stmt.startswith('param Q:='):
            vals = [int(x) for x in stmt.replace('param Q:=', '').split()]
            data['Q'] = {vals[i]: vals[i+1] for i in range(0, len(vals), 2)}
        elif stmt.startswith('param V:='):
            vals = [float(x) for x in stmt.replace('param V:=', '').split()]
            data['p'] = {i+1: vals[i] for i in range(len(vals))}
            data['R'] = list(data['p'].keys())
        elif stmt.startswith('param U:='):
            vals = [float(x) for x in stmt.replace('param U:=', '').split()]
            data['U'] = {0: 0.0}
            for i in range(len(vals)):
                data['U'][i+1] = vals[i]
        elif stmt.startswith('param qu :='):
            vals = [int(x) for x in stmt.replace('param qu :=', '').split()]
            data['q'] = {}
            data['I'] = {}
            i = 0
            while i < len(vals):
                node_id = vals[i]
                volume = vals[i+1]
                data['q'][node_id] = volume
                if node_id == data['Depot']:
                    data['I'][node_id] = 0
                    i += 2
                else:
                    data['I'][node_id] = vals[i+3]
                    i += 4
        elif stmt.startswith('param c[*,*]'):
            lines = stmt.split('\n')
            header_line = [l for l in lines if ':=' in l][0]
            col_ids = [int(x) for x in header_line.replace(':=', '').replace(':', '').split()]
            data['c'] = {}
            matrix_lines = [l.strip() for l in lines if l.strip() and ':=' not in l and 'param' not in l]
            for line in matrix_lines:
                parts = [int(x) for x in line.split()]
                row_id = parts[0]
                distances = parts[1:]
                for j, dist in enumerate(distances):
                    data['c'][(row_id, col_ids[j])] = dist

    return data

# ------------------------------------------
# 2. Load data and init parametrs
# ------------------------------------------
parser = argparse.ArgumentParser(description="Run MLP with input file")
parser.add_argument("filepath", type=str, help="Path to input file", nargs='?', default='instanciac10.mcgb',)
args = parser.parse_args()
filepath = args.filepath
print(f"Running with file: {filepath}")

data = parse_mcgb_file(filepath)

N = data['N']
N_farms = data['N_farms']
K = data['K']
R = data['R']
Depot = data['Depot']
c = data['c']
q = data['q']
I = data['I']
Q = data['Q']
U = data['U']
p = data['p']

# compartment config
num_compartments = 2
C = list(range(1, num_compartments + 1))
Q_c = {k: Q[k] / num_compartments for k in K} # capacitry per compartment

# Big M
M = max(Q.values()) * max(U.values())

# ------------------------------------------
# 3. Init Model
# ------------------------------------------
model = gp.Model("Multi_Compartment_Gradual_Blending")
model.Params.TimeLimit = 1200.0 # 20 minutes although set max timelimit to 14,400 (4 hrs) to be consistent with the paper
model.Params.LazyConstraints = 1

# variables
x = model.addVars(((i, j, k) for i in N for j in N if i != j for k in K), vtype=GRB.BINARY, name="x")
y = model.addVars(((i, k) for i in N for k in K), vtype=GRB.BINARY, name="y")

# add new constraint for farm milk distributed into specific compartments of truck k
v_farm = model.addVars(((i, k, c) for i in N_farms for k in K for c in C), vtype=GRB.CONTINUOUS, lb=0, name="v_farm")

# add a new index for blending variables by compartment 'c'
z = model.addVars(((k, c, r) for k in K for c in C for r in R), vtype=GRB.BINARY, name="z")
v = model.addVars(((k, c, r) for k in K for c in C for r in R), vtype=GRB.CONTINUOUS, lb=0, name="v")
w = model.addVars(((k, c, r) for k in K for c in C for r in R), vtype=GRB.CONTINUOUS, lb=0, name="w")

# ------------------------------------------
# 4. Obj and constraints
# ------------------------------------------
# (1) Obj: maximize profit (revenue from all compartments - routing cost)
revenue = gp.quicksum(p[r] * v[k, c, r] for k in K for c in C for r in R)
routing_cost = gp.quicksum(c[i, j] * x[i, j, k] for i in N for j in N if i != j for k in K)
model.setObjective(revenue - routing_cost, GRB.MAXIMIZE)

# (2) & (3) depot constraints
for k in K:
    model.addConstr(gp.quicksum(x[Depot, i, k] for i in N_farms) == 1, name=f"LeaveDepot_{k}")
    model.addConstr(gp.quicksum(x[i, Depot, k] for i in N_farms) == 1, name=f"ReturnDepot_{k}")

# (4) & (5) flow conservation
for k in K:
    for i in N_farms:
        model.addConstr(gp.quicksum(x[i, j, k] for j in N if j != i) == y[i, k], name=f"Leave_{i}_{k}")
        model.addConstr(gp.quicksum(x[j, i, k] for j in N if j != i) == y[i, k], name=f"Enter_{i}_{k}")

# (6) exact visits
for i in N_farms:
    model.addConstr(gp.quicksum(y[i, k] for k in K) == 1, name=f"VisitOnce_{i}")

# compartment loading logic
# if truck k visits farm i, the milk q[i] is split among compartments c
for i in N_farms:
    for k in K:
        model.addConstr(gp.quicksum(v_farm[i, k, c] for c in C) == q[i] * y[i, k], name=f"FarmSplit_{i}_{k}")

# (8) to (14) blending and volume constraints
# blending rules PER compartment
for k in K:
    for c in C:
        # a compartment gets AT MOST one grade (can be empty)
        model.addConstr(gp.quicksum(z[k, c, r] for r in R) <= 1, name=f"OneGrade_{k}_{c}")
        
        for r in R:
            model.addConstr(w[k, c, r] <= M * z[k, c, r], name=f"Link_w_z_{k}_{c}_{r}")
            model.addConstr(w[k, c, r] <= U[r] * v[k, c, r], name=f"UpperBoundSC_{k}_{c}_{r}")
            model.addConstr(w[k, c, r] >= U[r-1] * v[k, c, r] - M * (1 - z[k, c, r]), name=f"LowerBoundSC_{k}_{c}_{r}")
            # capacity is bounded by the compartment's specific capacity limit
            model.addConstr(v[k, c, r] <= Q_c[k] * z[k, c, r], name=f"VolCapacity_{k}_{c}_{r}")
            
        # total SC in compartment 'c' comes from the exact milk loaded into 'c'
        model.addConstr(gp.quicksum(w[k, c, r] for r in R) == 
                        gp.quicksum(v_farm[i, k, c] * I[i] for i in N_farms), name=f"TotalSC_{k}_{c}")
        
        # total volume in compartment 'c'
        model.addConstr(gp.quicksum(v[k, c, r] for r in R) == 
                        gp.quicksum(v_farm[i, k, c] for i in N_farms), name=f"TotalVol_{k}_{c}")

# (15) total volume collected must equal total volume produced constraint
model.addConstr(gp.quicksum(v[k, c, r] for k in K for c in C for r in R) == sum(q[i] for i in N_farms), name="TotalProducedCollected")


# ------------------------------------------
# 5. Subtour Elimination Callback
# ------------------------------------------
def subtour_elimination_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        x_vals = model.cbGetSolution(model._x)
        for k in K:
            edges = [(i, j) for i in N for j in N if i != j and x_vals[i, j, k] > 0.5]
            if not edges: continue
                
            adj = {i: [] for i in N}
            for i, j in edges: adj[i].append(j)
                
            visited = set()
            for start_node in N:
                if start_node in visited or start_node not in adj: continue
                component, stack = [], [start_node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        component.append(curr)
                        stack.extend([neighbor for neighbor in adj[curr] if neighbor not in visited])
                
                if Depot not in component and len(component) >= 2:
                    model.cbLazy(
                        gp.quicksum(model._x[i, j, k] for i in component for j in component if i != j) <= len(component) - 1
                    )
model._x = x

# ------------------------------------------
# 6. Optimize and Print Results
# ------------------------------------------
model.optimize(subtour_elimination_callback)

if model.status == GRB.OPTIMAL:
    print(f"\n--- OPTIMAL SOLUTION FOUND ---")
    print(f"Total Profit: ${model.ObjVal:.4f}\n")
    
    for k in K:
        print(f"--- Truck {k} ---")
        # route logic
        route_edges = [(i, j) for i in N for j in N if i != j and x[i, j, k].X > 0.5]
        if route_edges:
            current = Depot
            path, visited_farms = [Depot], []
            while True:
                next_nodes = [j for i, j in route_edges if i == current]
                if not next_nodes: break
                current = next_nodes[0]
                path.append(current)
                if current != Depot: visited_farms.append(current)
                if current == Depot: break
            print(f"Route: {' -> '.join(map(str, path))}")
            
            # print compartment breakdowns
            for c_id in C:
                print(f"  [Compartment {c_id}]")
                is_empty = True
                for r in R:
                    if z[k, c_id, r].X > 0.5:
                        is_empty = False
                        vol = v[k, c_id, r].X
                        sc_ratio = w[k, c_id, r].X / vol if vol > 0 else 0
                        print(f"    Assigned Grade: {r}")
                        print(f"    Total Volume: {vol:.3f} L (Capacity: {Q_c[k]})")
                        print(f"    Blend SC / ml: {sc_ratio:.3f}")
                        
                        # show which farms went into this compartment
                        for i in visited_farms:
                            farm_vol = v_farm[i, k, c_id].X
                            if farm_vol > 0.01:
                                print(f"      <- {farm_vol:.2f} L from Farm {i} (Original SC: {I[i]})")
                if is_empty:
                    print(f"    (Empty)")
        print()
else:
    print("No optimal solution found.")