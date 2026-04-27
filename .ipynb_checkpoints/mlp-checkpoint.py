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

    content = re.sub(r'#.*', '', content) # remove comments
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
            # maps truck_id to capacity
            data['Q'] = {vals[i]: vals[i+1] for i in range(0, len(vals), 2)}
            
        elif stmt.startswith('param V:='):
            vals = [float(x) for x in stmt.replace('param V:=', '').split()]
            # revenue per grade (1, 2, 3, etc)
            data['p'] = {i+1: vals[i] for i in range(len(vals))}
            data['R'] = list(data['p'].keys())
            
        elif stmt.startswith('param U:='):
            vals = [float(x) for x in stmt.replace('param U:=', '').split()]
            # upper bounds per grade. U[0] = 0 is set as the default
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
                    # farm has 4 values: id, volume, type (ignored for blending output), and SC per ml
                    sc_per_ml = vals[i+3]
                    data['I'][node_id] = sc_per_ml
                    i += 4
                    
        elif stmt.startswith('param c[*,*]'):
            # get matrix of values to solve on
            lines = stmt.split('\n')
            header_line = [l for l in lines if ':=' in l][0]
            print(header_line)
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

# total somatic cells at farm i (Volume * SC per ml)
e = {i: q[i] * I[i] for i in N}

M = max(Q.values()) * max(U.values())

# ------------------------------------------
# 3. Init Model
# ------------------------------------------
model = gp.Model("Milk_Collection_Gradual_Blending")
model.Params.TimeLimit = 1200.0 # 20 minutes although set max timelimit to 14,400 (4 hrs) to be consistent with the paper
model.Params.LazyConstraints = 1

# variables
x = model.addVars(((i, j, k) for i in N for j in N if i != j for k in K), vtype=GRB.BINARY, name="x")
y = model.addVars(((i, k) for i in N for k in K), vtype=GRB.BINARY, name="y")
z = model.addVars(((k, r) for k in K for r in R), vtype=GRB.BINARY, name="z")
v = model.addVars(((k, r) for k in K for r in R), vtype=GRB.CONTINUOUS, lb=0, name="v")
w = model.addVars(((k, r) for k in K for r in R), vtype=GRB.CONTINUOUS, lb=0, name="w")

# ------------------------------------------
# 4. Obj and constraints
# ------------------------------------------
# (1) Obj: maximize profit
revenue = gp.quicksum(p[r] * v[k, r] for k in K for r in R)
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

# (8) to (14) blending and volume constraints
for k in K:
    for r in R:
        model.addConstr(w[k, r] <= M * z[k, r], name=f"Link_w_z_{k}_{r}")
        model.addConstr(w[k, r] <= U[r] * v[k, r], name=f"UpperBoundSC_{k}_{r}")
        model.addConstr(w[k, r] >= U[r-1] * v[k, r] - M * (1 - z[k, r]), name=f"LowerBoundSC_{k}_{r}")
        model.addConstr(v[k, r] <= Q[k] * z[k, r], name=f"VolCapacity_{k}_{r}")
        
    model.addConstr(gp.quicksum(z[k, r] for r in R) == 1, name=f"OneGrade_{k}")
    model.addConstr(gp.quicksum(w[k, r] for r in R) == gp.quicksum(e[i] * y[i, k] for i in N_farms), name=f"TotalSC_{k}")
    model.addConstr(gp.quicksum(v[k, r] for r in R) == gp.quicksum(q[i] * y[i, k] for i in N_farms), name=f"TotalVol_{k}")

# (15) total volume collected must equal total volume produced constraint
model.addConstr(gp.quicksum(v[k, r] for k in K for r in R) == sum(q[i] for i in N_farms), name="TotalProducedCollected")

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
            for i, j in edges:
                adj[i].append(j)
                
            visited = set()
            for start_node in N:
                if start_node in visited or start_node not in adj: continue
                
                component = []
                stack = [start_node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        component.append(curr)
                        stack.extend([neighbor for neighbor in adj[curr] if neighbor not in visited])
                
                # check if component is an isolated cycle (Depot is not in it)
                if Depot not in component and len(component) >= 2:
                    model.cbLazy(
                        gp.quicksum(model._x[i, j, k] for i in component for j in component if i != j) 
                        <= len(component) - 1
                    )

model._x = x

# ------------------------------------------
# 6. Run optimization and print results
# ------------------------------------------
model.optimize(subtour_elimination_callback)

if model.status == GRB.OPTIMAL:
    print(f"\n--- OPTIMAL SOLUTION FOUND ---")
    print(f"Total Profit: ${model.ObjVal:.4f}\n")
    
    for k in K:
        print(f"Truck {k}:")
        
        for r in R:
            if z[k, r].X > 0.5:
                print(f"  Assigned Grade: {r}")
                print(f"  Total Volume: {v[k, r].X:.3f} L")
                # calculate the actual SC per ml ratio for the blend
                sc_ratio = w[k, r].X / v[k, r].X if v[k, r].X > 0 else 0
                print(f"  Blend SC / ml: {sc_ratio:.3f} (Limit: {U[r-1]} < SC <= {U[r]})")
                
        route_edges = [(i, j) for i in N for j in N if i != j and x[i, j, k].X > 0.5]
        if route_edges:
            current = Depot
            path = [Depot]
            while True:
                next_nodes = [j for i, j in route_edges if i == current]
                if not next_nodes: break
                current = next_nodes[0]
                path.append(current)
                if current == Depot: break
            print(f"  Route: {' -> '.join(map(str, path))}\n")
else:
    print("No optimal solution found.")