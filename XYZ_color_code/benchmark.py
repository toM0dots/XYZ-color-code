import numpy as np
import matplotlib.pyplot as plt
import math
import random
import multiprocessing as mp
import csv
import json
import argparse
# initialize the codespace with only A-stabilizer 

def lattice_space(H, L):
    lattice = np.zeros((H, L))
    for i in range(H):
        for j in range(L):
            if i % 3 == 0 and j % 3 != 0:
                lattice[i][j] = 1
            if i % 3 == 1 and j % 3 != 2:
                lattice[i][j] = 1
            if i % 3 == 2 and j % 3 != 1:
                lattice[i][j] = 1
    return lattice
import random

def energy(lattice):
    
    H = lattice.shape[0]
    L = lattice.shape[1]
    energy = 0
    for i in range(H):
        for j in range(L):
            # Periodic boundary conditions
            right = lattice[(i+1)% H, (j+1) % L]
            down  = lattice[(i+1) % H, j]
            energy +=  (lattice[i, j] + right + down)% 2
    return energy

def compute_dE(lattice, i, j):
    # Sum over four neighbors with periodic boundaries
    H = lattice.shape[0]
    L = lattice.shape[1]
    s = lattice[i, j]
    ds = (s+1)%2
    left = lattice[i, (j-1)%L] + lattice [(i-1)%H, (j-1)%L]
    right = lattice[(i-1)%H, j] + lattice[i, (j+1)%L]
    bottom = lattice[(i+1)%H, j] + lattice[(i+1)%H, (j+1)%L]
    E  = (left+s)%2 + (right+s)%2 + (bottom+s)%2
    E_new = (left+ds)%2 + (right+ds)%2 + (bottom+ds)%2
    dE = E_new - E
    if dE == 0:
        raise ValueError("No change in energy")
    return dE

def monte_carlo_error(lattice, num_steps, error_rate):
    H = lattice.shape[0]
    L = lattice.shape[1]
    for i in range(H):
        for j in range(L):
            r = random.random()
            if r < error_rate:
                # Flip the qubit
                lattice[i, j] = (lattice[i, j] + 1) % 2
            """
        for i in range(H):
            for j in range(L):
                r = random.random()
                if r < error_rate:
                    # Flip the qubit
                    lattice[i, j] = (lattice[i, j] + 1) % 2
            """

    return lattice
    
def nonzero_random():
    r = random.random()
    while r == 0.0:
        r = random.random()
    return r

def bkl(lattice, steps, error_rate, energy_hist=[], time_list=[], spin_list=[], mag_list = [], nw = [], sw= [], correction = True):
    f = 1e-8
    beta = math.log((3+error_rate)/error_rate)/3
    H = lattice.shape[0]
    L = lattice.shape[1]
    # mag_list =[]
    for _ in range(steps):
        # for i_ in range(H):
        #     for j_ in range(L):
        #         r = random.random()
        #         if r < error_rate:
        #             # Flip the qubit
        #             lattice[i_, j_] = (lattice[i_, j_] + 1) % 2

        # Create a dictionary to group spin flips by their energy change (dE)
        classes = {}
        # For each spin, compute its energy change upon flipping
        for i in range(H):
            for j in range(L):
                dE = compute_dE(lattice, i, j)
                if dE not in classes:
                    classes[dE] = []
                classes[dE].append((i, j))
        row = np.array([len(classes.get(-3, [])), len(classes.get(-1, [])), len(classes.get( 1, [])), len(classes.get( 3, []))])
        # for dE in (-3, -1, 1, 3):
        #     classes.setdefault(dE, [])
        nw.append(row)
        # Calculate transition rates for each class:
        # For spins where flipping lowers energy (dE <= 0), acceptance probability is 1;
        # for dE > 0, it is dE*(1-exp(-beta * dE))
        rates = {}
        total_rate = 0.0
        for dE in sorted(classes):
            if correction == True:
                rate = len(classes[dE])* (dE / (np.exp(beta * dE)-1 ) )#- error_rate)
            else:
                rate = error_rate
            rates[dE] = rate
            total_rate += rate
        if total_rate == 0:
            print(len(classes), "total_rate is 0")
            continue
        rate_row = np.array([rates.get(-3, 0.0), rates.get(-1, 0.0), rates.get(1, 0.0), rates.get(3, 0.0)])
        sw.append(rate_row)
        # rate_list = []
        # rate_list.append(total_rate)
        # Choose a class according to the rates (weighted selection)
        r = random.uniform(0, total_rate)
        cumulative = 0.0
        chosen_class = None
        for dE, rate in rates.items():
            cumulative += rate
            if r <= cumulative :
                # print("hi")
                chosen_class = dE
                break
        if total_rate == 0.0:
            raise ValueError("Total rate is zero, cannot choose a class.")
        spin_list.append(chosen_class)   
        
        # From the chosen class, select a spin uniformly at random
        i, j = random.choice(classes[chosen_class])
        # Flip the spin
        lattice[i, j] = (lattice[i, j] + 1) % 2
        energy_hist.append(energy(lattice))   
        rho3 = nonzero_random()
        dT = -1/total_rate * np.log(rho3)
        # print(rho3, total_rate)
        
        time_list.append(dT)
        mag_list.append(np.sum(lattice))
    return lattice



# optimal decoder
import numpy as np

def syndrome_detector(lattice):
    """
    Detects errors in syndrom measuring A-stabilizer
    """
    H = lattice.shape[0]
    L = lattice.shape[1]
    errors = np.zeros((H,L), dtype=int)
    #errors = np.zeros((H,L))
    for i in range(H):
        for j in range(L):
            if (lattice[i, j]+lattice[(i+1) %H, j]+lattice[(i+1)%H, (j+1)%L]) % 2 != 0:  # error by A-stablizer
                #errors[i][j]= 1
                errors[i][j] = 1  
    return errors

def error_qubit_detector(lattice):
    """
    Detects errors in the lattice and returns a list of qubits to be flipped.
    """
    H = lattice.shape[0]
    L = lattice.shape[1]
    errors = syndrome_detector(lattice)
    error_qubits = np.zeros((H,L),dtype=int)
    for i in range(H):
        for j in range(L):
            if not (errors[i][j] or errors[(i-1)%H][j] or errors[(i-1)%H][(j-1)%L]):  
                error_qubits[i][j]= 1   # Assuming 1 indicates an error
    return error_qubits

def sweeper_test(lattice):
    """
    Sweeps through the lattice and move the defects. to the first row.
    """
    H = lattice.shape[0]
    L = lattice.shape[1]
    errors = np.zeros((H,L))
    latt = lattice.copy()
    for i in range(H-1):      # to H or H-1?/
        for j in range(L):
            if (latt[(H-i-1)%H, j]+latt[(H-i)%H, j]+latt[(H-i)%H, (j+1)%L]) % 2 == 1:  # error by A-stablizer
                errors[(H-i-1)%H][j]= 1
                latt[(H-i-1)% H][j] = (latt[(H-i-1) % H][j] + 1) % 2 
    return latt, errors
def sweeper(lattice):
    H = lattice.shape[0]
    L = lattice.shape[1]
    errors = np.zeros((H,L), dtype=int)
    latt = lattice.copy()
    for i in range(H-1):      # to H or H-1?/
        for j in range(L):
            if latt[(H-i-1)%H, j] == 1:
                errors[(H-i-1)%H][j]= 1
                latt[(H-i-1)% H][j] = (latt[(H-i-1) % H][j] + 1) % 2
                latt[(H-i-2)% H][j] = (latt[(H-i-2) % H][j] + 1) % 2
                latt[(H-i-2)% H][(j-1) % L] = (latt[(H-i-2) % H][(j-1) % L] + 1) % 2
    return latt, errors


def update_rule(lattice):
    """
    Update rule 102, input row_i+1, returns row_i
    """
    # H = lattice.shape[0]
    L = lattice.shape[1]
    f = np.eye(L,dtype=int)
    f += cyclic_shift(L)  # cyclic shift matrix
    # for j in range(L):
    #     for i in range(L):
    #         if j % 3 ==0 and i % 3 == 0:
    #             f[j][i] = 1
    #         if j % 3 == 0 and i % 3 == 1:
    #             f[j][i] = 1
    #         if j % 3 == 1 and i % 3 == 1:
    #             f[j][i] = 1
    #         if j % 3 == 1 and i % 3 == 2:
    #             f[j][i] = 1
    #         if j % 3 == 2 and i % 3 == 0:
    #             f[j][i] = 1
    #         if j % 3 == 2 and i % 3 == 2:
    #             f[j][i] = 1
    
    return f

def cyclic_shift(L):
    """
    Return the L×L matrix P which, when you do P @ v,
    cyclically shifts the entries of v up by one:

       (Pv)[i] = v[(i+1) % L].
    """
    P = np.zeros((L, L), dtype=int)
    for i in range(L):
        P[i, (i+1) % L] = 1
    return P

def invert_mod2(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a square binary matrix A over GF(2).
    
    Parameters
    ----------
    A : (n,n) array_like of {0,1}
        The matrix to invert; must be square and of full rank mod 2.
    
    Returns
    -------
    A_inv : (n,n) ndarray of {0,1}
        The inverse of A in Z_2, so that A_inv @ A ≡ I mod 2.
    
    Raises
    ------
    ValueError
        If A is not square or not invertible over Z_2.
    """
    A = np.array(A, dtype=np.uint8) & 1
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")
    
    # Build the augmented matrix [A | I]
    aug = np.concatenate([A, np.eye(n, dtype=np.uint8)], axis=1)  # shape (n, 2n)
    
    # Gauss–Jordan elimination
    row = 0
    for col in range(n):
        # 1) Find a pivot in or below current row
        pivot = None
        for r in range(row, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None:
            # no pivot => singular
            raise ValueError("Matrix is singular in GF(2), cannot invert")
        
        # 2) Swap pivot row into place if needed
        if pivot != row:
            aug[[row, pivot], :] = aug[[pivot, row], :]
        
        # 3) Eliminate all other 1’s in this column
        for r in range(n):
            if r != row and aug[r, col]:
                # row r ← row r XOR row 'row'
                aug[r, :] ^= aug[row, :]
        
        row += 1
    
    # At this point the left n×n block is I, so the right block is A^{-1}
    A_inv = aug[:, n:]  # shape (n, n)
    A_inv = A_inv.astype(np.uint8) & 1  # ensure binary
    return A_inv

def optimal_decoder(lattice):
    """
    Optimal decoder using sweeper. # treat S as column vector
    """
    H = lattice.shape[0]
    L = lattice.shape[1]
    # lattice1, _ = sweeper(lattice)
    # print("lattice", lattice)
    S = np.zeros((1,L), dtype=int)
    for i in range(L):
        if (lattice[0][i] + lattice[1][i]+lattice[1][(i+1)%L]) % 2 == 1:
            S[0][i] = 1
        # S[i] = lattice[0, i]
    S = np.transpose(S)  
    # print("S", S)
    # S = np.transpose(lattice[0])
    f = update_rule(lattice)
    f1 = np.linalg.matrix_power(f, H-1) % 2
    # print(f)
    P = cyclic_shift(L)
    # f1 = f.copy() 
    # for i in range(L):
    #     f1[:,(i+1) % L] = f[:,i] #######################. is it i+1 or i-1?

    A = (np.eye(L) + f1 + P @ f1) % 2
    # print("A", A)
    E, _ = solve_mod2(A, S)
    # A_inv = invert_mod2(A)
    # E = (A_inv @ S ) % 2
    # E = np.transpose(E)
    # diff = E - S
    return E
    # return e_int, S, E, diff

def C2_constructor(E, lattice):
    """
    Construct the C2 code from the error syndrome in line 0. Again E is the first row written as column vector.
    """
    H = lattice.shape[0]
    L = lattice.shape[1]
    C2 = np.zeros((L, H))
    E = E.squeeze()  # Ensure E is a 1D array
    f = update_rule(lattice)
    for i in range(H):
        C2[:,(H-i)%H] = (np.linalg.matrix_power(f, i) & 1 ) @ E
    C2 = np.transpose(C2)
    return C2

def logical_operators(lattice):
    H = lattice.shape[0]
    L = lattice.shape[1]
    Lattice = np.array(lattice_space(H, L))
    f = update_rule(Lattice)
    lattice1 = np.transpose((f @ np.transpose(Lattice )) % 2)
    lattice2 = (Lattice + lattice1) % 2
    lattice3 = np.zeros((H, L))
    logicals = [Lattice, lattice1, lattice2, lattice3]      # 1. L + M 2. M  3. L 4.vaccum
    return logicals

def logical_checks(lattice, x):
    """
    Optimal decoder using sweeper.
    """
    H = lattice.shape[0]
    L = lattice.shape[1]
    R = logical_operators(lattice)
    # syndromes = syndrome_detector(lattice)
    latt = lattice.copy()
    
    syndrome, C1 = sweeper_test(latt)
    # print("C1", C1)
    # print("syndrome", syndrome)
    E = optimal_decoder(syndrome)
    C2 = C2_constructor(E, latt)
    # print("C2", C2)
    weights = [( (R_i + C1 + C2) % 2 ).sum() for R_i in R]
    min_weight = min(weights)
    min_index = weights.index(min_weight)
    check = ((R[min_index] + C1 + C2 + lattice + R[x]) % 2 ).sum()
    # print(weights)
    # print(check)
    if check == 0:
        return True
    else:
        return False
    print(weights)
  
    if min_index != x:
        return False #min_index
    else:
        return True

def solve_mod2(A, y):
    """
    Solve y = A x over GF(2).  Returns (x_particular, nullspace_basis).
    """
    A = np.array(A, dtype=np.uint8) & 1
    y = np.array(y, dtype=np.uint8).flatten() & 1
    r, c = A.shape

    # Build [A | y]
    aug = np.concatenate([A.copy(), y.reshape(-1,1)], axis=1)

    pivot_cols = []
    row = 0

    # Forward elimination in GF(2)
    for col in range(c):
        # Find a row i≥row with aug[i,col] == 1
        pivot = None
        for i in range(row, r):
            if aug[i, col] == 1:
                pivot = i
                break
        if pivot is None:
            continue  # free column

        # Swap that pivot row into position `row`
        if pivot != row:
            aug[[row, pivot], :] = aug[[pivot, row], :]

        # Eliminate all other 1’s in this column by XOR
        for i in range(r):
            if i != row and aug[i, col] == 1:
                aug[i, :] ^= aug[row, :]

        pivot_cols.append(col)
        row += 1
        if row == r:
            break

    # Check for inconsistency: any row [0 … 0 | 1] means no solution
    for i in range(row, r):
        if np.all(aug[i, :c] == 0) and aug[i, c] == 1:
            raise ValueError("No solutions exist for A x = y (mod 2).")

    # Identify free variables
    pivot_set = set(pivot_cols)
    free_cols = [col for col in range(c) if col not in pivot_set]

    # Build one particular solution: set all free x[j]=0
    x_part = np.zeros(c, dtype=np.uint8)
    for i, pcol in enumerate(pivot_cols):
        x_part[pcol] = aug[i, c]

    # Build nullspace basis (not needed here, but returned for completeness)
    nullspace_basis = []
    for f in free_cols:
        x_hom = np.zeros(c, dtype=np.uint8)
        x_hom[f] = 1
        for i, pcol in enumerate(pivot_cols):
            if aug[i, f] == 1:
                x_hom[pcol] = 1
        nullspace_basis.append(x_hom)

    return x_part, nullspace_basis


import multiprocessing as mp

def run_until_truncation(seed: int,
                         logical: int,
                         error_rate: float,
                         H: int,
                         L: int) -> tuple[float, list[float], list[float]]:
    random.seed(seed)
    beta = math.log((3 + error_rate) / error_rate) / 3
    time_list = []
    energy_list = []
    mag_list =[]
    x = lattice_space(H, L)
    logicals = logical_operators(x)
    lattice = logicals[logical].copy()
    step = 0
    while True:
        step += 1
        lattice = bkl(lattice, 1, energy_list, error_rate, time_list)
        mag_list.append(lattice.sum())
        # lattice = monte_carlo_error(lattice, 1 , error_rate)
        if step % 10 == 0 and not logical_checks(lattice, logical):
            break
    mem_time = sum(time_list)
    return mem_time, energy_list,  mag_list
#    mem_times.append(list(mem_time))
#    energies.append(list(energy_list))
#    mags.append(list(mag_list))
    

def run_until_truncation_test(seed: int,
                              logical: int,
                         error_rate: float,
                         out_list: list):
    random.seed(seed)
    beta = math.log((3 + error_rate) / error_rate) / 3
    x = lattice_space(9, 6)
    logicals = logical_operators(x)
    lattice = logicals[logical].copy()
    # lattice = bkl(lattice, 1, [], error_rate, [])
    lattice = monte_carlo_error(lattice, 1 , error_rate)
    flag = logical_checks(lattice, logical)
    out_list.append(flag)
    


def run_until_truncation_mag(seeds: int,
                         steps: int,
                         logical: int,
                         error_rate: float,
                         H: int,
                         L: int):
    
    E = H * L

    num_runs  = seeds
    num_steps = steps

    # ------------------------------------------
    # 1) Collect all decoder (bkl) trajectories
    # ------------------------------------------
    decoder_trajs = np.zeros((num_runs, num_steps))
    decoder_times = np.zeros((num_runs, num_steps))  # if bkl populates it; otherwise you can drop it
    decoder_mag = np.zeros((num_runs, num_steps))
    mem_times = []
    for run_idx, seed in enumerate(range(num_runs)):
        random.seed(seed)
        x = lattice_space(H, L)
        logicals = logical_operators(x)
        lattice = logicals[logical].copy()
        

        energy_list = []
        time_list   = []            # if bkl populates it; otherwise you can drop it
        mags = []
        for i in range(num_steps):
            lattice = bkl(lattice, 1, error_rate, energy_list,  time_list, mag_list=mags)
            if i % 20 == 0 and not logical_checks(lattice, logical):
                mem_times.append(np.sum( time_list))
                

        if len(energy_list) != num_steps:
            raise ValueError(f"Run {run_idx}: expected {num_steps} energies, got {len(energy_list)}")

        decoder_trajs[run_idx, :] = energy_list
        decoder_times[run_idx, :] = time_list  # if bkl populates it; otherwise you can drop it
        decoder_mag[run_idx, :] = mags  # Store the magnetization trajectory
    # compute average (and normalize)
    mean_energy = decoder_trajs.mean(axis=0) / E
    std_decoder = decoder_trajs.std(axis=0) / E
    mean_times = decoder_times.mean(axis=0)  # dT list  
    mean_times_cum = np.cumsum(mean_times)  # Cumulative sum of time steps
    mean_mag = decoder_mag.mean(axis=0) /E  # Store the average magnetization trajectory
    mean_mem = np.mean(mem_times)
    # mean_step = np.mean(mem_times[:,0])

    # x = np.arange(5000)
    plt.figure(figsize=(10, 3))
    plt.plot(mean_times_cum, mean_energy, color ='blue', label='energy')
    plt.plot(mean_times_cum, mean_mag, color ='green', label='magnitization')
    plt.axvline(x=mean_mem, color='red', linestyle='--', linewidth=2, label= f'{mean_mem}')
    # plt.plot(x, bar_list, color ='red',  label='2000 step errors without decoding')
    plt.xlabel('physical times')
    plt.ylabel('Energy')
    plt.title(f'Energy vs Steps on Lattice {H}x{L} error {error_rate}')
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/E(t)..M(t)_lattice{H}*{L}_error_{error_rate}.png')
    plt.show()
    

    # plt.figure(figsize=(10, 3))
    # plt.plot(np.arange(len(mean_times)), mean_energy, color ='blue', label='energy')
    # plt.plot(np.arange(len(mean_times)), mean_mag, color ='green', label='magnitization')
    # plt.axvline(x=mean_step, color='red', linestyle='--', linewidth=2, label= f'{mean_mem}')
    # # plt.plot(x, bar_list, color ='red',  label='2000 step errors without decoding')
    # plt.xlabel('MC steps')
    # plt.ylabel('Energy')
    # plt.title(f'Energy vs Steps on Lattice {H}x{L} error {error_rate}')
    # plt.legend()
    # plt.grid()
    # plt.savefig(f'E(t)..M(t)_lattice{H}*{L}_error_{error_rate}_MC.png')
    # plt.show()
    

def main():
    parser = argparse.ArgumentParser(description='Run memory time simulation.')

    # parser.add_argument('--seeds', type=int, nargs='+', default= list(range(100)),
    #                     help='Random seed for reproducibility.')
    
    parser.add_argument('--seeds', type=int,  default= 100,
                        help='Random seed for reproducibility.')
    parser.add_argument('--logical', type=int, default=3,
                        help='Index of logical operator to test (default: 3, all up).')
    
    parser.add_argument('--error_rate', type=float, default=  0.0001,
                        help='Physical Error rate for the BKL update.')

    parser.add_argument('--H', type=int, default=12,
                        help='Lattice Height.')

    parser.add_argument('--L', type=int, default=9,
                        help='Lattice Length.')
    parser.add_argument('--steps', type=int, default=1000,
                        help='simulation steps.')
                        
    args = parser.parse_args()
    seeds = args.seeds
    logical = args.logical
    error_rate = args.error_rate
    H = args.H
    L = args.L
    steps = args.steps
    run_until_truncation_mag(seeds, steps, logical, error_rate, H, L)

# def main():
def main_main():
    parser = argparse.ArgumentParser(description='Run memory time simulation.')

    parser.add_argument('--seeds', type=int, nargs='+', default= list(range(100)),
                        help='Random seed for reproducibility.')
    
    parser.add_argument('--logical', type=int, default=3,
                        help='Index of logical operator to test (default: 3, all up).')
    
    parser.add_argument('--error_rate', type=float, default=  0.0001,
                        help='Physical Error rate for the BKL update.')

    parser.add_argument('--H', type=int, default=12,
                        help='Lattice Height.')

    parser.add_argument('--L', type=int, default=9,
                        help='Lattice Length.')
                        
    args = parser.parse_args()
    seeds = args.seeds
    logical = args.logical
    error_rate = args.error_rate
    H = args.H
    L = args.L
    


    work_args = [
        (s, logical, error_rate, H, L)
        for s in seeds
    ]

    # run in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(run_until_truncation, work_args)
        # each result is (mem_time: float, energies: list[float], mags: list[float])

    # unzip the results
    mem_times, energies, mags = map(list, zip(*results))

    # write CSV with JSON-encoded lists
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['seed', 'mem_time', 'energies', 'mags'])
        for seed, mt, en, mg in zip(args.seeds, mem_times, energies, mags):
            writer.writerow([
                seed,
                f"{mt:.6f}",
                json.dumps(en),
                json.dumps(mg),
            ])

    print(f"Wrote results.csv with {len(args.seeds)} rows.")

    mem_times = list(mem_times)
    
    if not mem_times:
        raise RuntimeError("No results collected—check your worker function!")
    avg_mem_times = sum( mem_times) / (len( mem_times))
    max_mem_times = max( mem_times)
    # errors = stop_steps.count(False)/len(stop_steps)
    # print("Logical Error for each seed:", stop_steps)
    # print("logical error rate:", errors)
    print("Memory time for each seed:",  mem_times)
    print(f'Average stopping step: { avg_mem_times:.2f}')
    print("Max Stop:", max_mem_times)



    
if __name__ == '__main__':
    main()
