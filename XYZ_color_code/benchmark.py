import numpy as np
import matplotlib.pyplot as plt
import math
import random
import multiprocessing as mp
import csv
import json
import argparse
import copy

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
    r = 1.0 - random.random()
    while r == 0.0:
        r = 1.0- random.random()
    return r

def bkl(lattice, steps, error_rate, energy_hist= None, time_list= None, spin_list= None, nw = None, sw= None, correction = True, debug = False, config=False):
    if energy_hist is None: energy_hist = []
    if time_list   is None: time_list   = []
    if spin_list   is None: spin_list   = []
    if nw          is None: nw          = []
    if sw          is None: sw          = []

    configs     = []
    total_R = []
    mag_list = []
    f = 1e-8
    beta = math.log((3+error_rate)/error_rate)/3
    H = lattice.shape[0]
    L = lattice.shape[1]
    # mag_list =[]
    spin_class_check = []
    for step in range(steps):

        # Create a dictionary to group spin flips by their energy change (dE)
        classes = {-3:[], -1:[], 1:[], 3:[]}
        # For each spin, compute its energy change upon flipping
        for i in range(H):
            for j in range(L):
                dE = compute_dE(lattice, i, j)
                if dE == -3:
                    classes[-3].append((i, j))
                elif dE == -1:
                    classes[-1].append((i, j))
                elif dE == 1:
                    classes[ 1].append((i, j))
                elif dE == 3:
                    classes[ 3].append((i, j))
                # classes[dE].append((i, j))
        row = np.array([len(classes.get(-3, [])), len(classes.get(-1, [])), len(classes.get( 1, [])), len(classes.get( 3, []))])
        
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
            
        # Choose a class according to the rates (weighted selection)
        r = np.random.uniform(0, total_rate)
        cumulative = 0.0
        chosen_class = None
        for dE in sorted(rates):
            cumulative += rates[dE]
            if r <= cumulative :
                chosen_class = dE
                break
        
        E_prev = energy(lattice)
        # From the chosen class, select a spin uniformly at random
        i, j = random.choice(classes[chosen_class]) 
        # Flip the spin
        lattice[i, j] = (lattice[i, j] + 1) % 2 
        E_new = energy(lattice)

        if total_rate == 0.0:
            raise ValueError("Total rate is zero, cannot choose a class.")
        
        
        if len(classes[chosen_class]) == 0 or chosen_class not in classes: 
            raise ValueError("Chosen class has no spins to flip.")
        
        if debug:
            # spin_class_check.append(copy.deepcopy(classes))
            nw.append(row)
            rate_row = np.array([rates.get(-3, 0.0), rates.get(-1, 0.0), rates.get(1, 0.0), rates.get(3, 0.0)])
            sw.append(rate_row)
            spin_list.append(chosen_class)   
            configs.append(copy.deepcopy(lattice))
            total_R.append(total_rate)
        

        # ============= spin flip check ==========
            check = np.zeros((H,L))
            check[i,j] = 1
            if len(configs)>0 and not np.array_equal(((lattice + check) % 2) , configs[-1] ) :
                print('i,j', i, j)
                print('current config\n', lattice)
            
                print('last config\n', configs[-1])
                print('step', step)
                raise ValueError('Spin not flipped correctly')
            

        # ============= energy check ==========
            if E_new != E_prev + chosen_class:
                print('current energy', energy_hist[-1] )
                print('dE', chosen_class)
                print('last enregy', energy_hist[-2])
                print('spin config\n', lattice)
                print('i,j', i, j)
                print('step', len(energy_hist))
                print('energ list', energy_hist)
                print('rate', rates)
                print('previsou config\n', configs[-2])
                print('current config\n', configs[-1])
                print('before before config\n', configs[-3])
                raise ValueError('Energy not consistent')  
         
    # ============= rate check ============  
        # if len(rate)


        rho3 = nonzero_random()
        dT = 1/total_rate * -np.log(rho3)

        if config:
            energy_hist.append(energy(lattice))
            time_list.append(dT)
            mag_list.append(np.sum(lattice) / (H * L))
    if debug:
        out = {'spin_class_check': spin_class_check, 'nw': nw, 'sw': sw, 'spin_list': spin_list}
        return out
    elif config:
        out = {'lattice': lattice, 'energy_hist': energy_hist, 'time_list': time_list, 'mag_list': mag_list}
        return out
    else:
        return lattice


def thermal_decoder_cached_step(
    lattice,
    error_rate,
    b=None,
    dE_array=None,
    weight_array=None,
    total_weight=None,
    flags=None,
    beta=None,
    weight_rule="negative_boltzmann",
    rng=None,
    debug_check=False,
):
    """
    Cached one-step zero-temperature-like thermal decoder.

    negative_boltzmann rule:
        weight = exp(-beta * dE) if dE < 0
        weight = 0               if dE >= 0

    Uses diagonal Newman-Moore geometry:
        spin (i,j) touches:
            b[i,j], b[i-1,j], b[i-1,j-1]
    """

    import numpy as np
    import math

    if rng is None:
        rng = np.random.default_rng()

    H, L = lattice.shape

    if beta is None:
        beta = math.log((3 + error_rate) / error_rate) / 3

    if flags is None:
        flags = np.zeros((H, L), dtype=bool)
    else:
        flags = np.asarray(flags, dtype=bool)

    # ============================================================
    # 1. Initialize cache if needed
    # ============================================================

    if b is None:
        b = compute_b_nm(lattice).astype(np.uint8)
    else:
        b = b.astype(np.uint8, copy=False)

    if dE_array is None or weight_array is None or total_weight is None:

        bb = b.astype(np.int16)

        # Diagonal convention:
        # dE(i,j) depends on b[i,j], b[i-1,j], b[i-1,j-1]
        touched = (
            bb
            + np.roll(bb, shift=1, axis=0)
            + np.roll(np.roll(bb, shift=1, axis=0), shift=1, axis=1)
        )

        dE_array = (3 - 2 * touched).astype(np.int8)

        dE_float = dE_array.astype(float)

        if weight_rule == "boltzmann":
            dE_float = dE_array.astype(float)
            weight_array = np.exp(-beta * dE_float)

        elif weight_rule == "negative_boltzmann":
            weight_array = np.where(
                dE_array < 0,
                np.exp(-beta * dE_float),
                0.0,
            )

        elif weight_rule == "metropolis":
            weight_array = np.where(
                dE_array <= 0,
                1.0,
                np.exp(-beta * dE_float),
            )

        elif weight_rule == "bkl":
            denom = np.exp(beta * dE_float) - 1.0
            weight_array = np.zeros_like(dE_float, dtype=float)

            nonzero = np.abs(denom) > 1e-14
            weight_array[nonzero] = dE_float[nonzero] / denom[nonzero]
            weight_array[~nonzero] = 1.0 / beta

        else:
            raise ValueError(
                "weight_rule must be 'boltzmann', 'negative_boltzmann', 'metropolis', or 'bkl'."
            )

        weight_array[flags] = 0.0
        total_weight = float(weight_array.sum())

    # ============================================================
    # 2. No allowed thermal move
    # ============================================================

    if total_weight <= 0:
        return {
            "lattice": lattice,
            "b": b,
            "dE_array": dE_array,
            "weight_array": weight_array,
            "total_weight": total_weight,
            "chosen_spin": None,
            "chosen_dE": None,
            "beta": beta,
        }

    # ============================================================
    # 3. Choose spin by cached weights
    # ============================================================

    prob_flat = weight_array.ravel() / total_weight
    flat_index = rng.choice(H * L, p=prob_flat)

    i, j = np.unravel_index(flat_index, (H, L))

    i = int(i)
    j = int(j)

    chosen_spin = (i, j)
    chosen_dE = int(dE_array[i, j])

    if debug_check:
        E_before = energy(lattice)

    # ============================================================
    # 4. Flip chosen spin
    # ============================================================

    lattice[i, j] ^= 1

    # ============================================================
    # 5. Toggle affected syndrome bits
    # ============================================================

    # Diagonal geometry:
    # spin (i,j) toggles b[i,j], b[i-1,j], b[i-1,j-1]
    toggled_stabs = [
        (i % H, j % L),
        ((i - 1) % H, j % L),
        ((i - 1) % H, (j - 1) % L),
    ]

    for a, c in toggled_stabs:
        b[a, c] ^= 1

    # ============================================================
    # 6. Find spins whose dE values changed
    # ============================================================

    affected = set()

    # If stabilizer b[a,c] changes, it affects spins:
    # (a,c), (a+1,c), (a+1,c+1)
    for a, c in toggled_stabs:
        affected.add((a % H, c % L))
        affected.add(((a + 1) % H, c % L))
        affected.add(((a + 1) % H, (c + 1) % L))

    # ============================================================
    # 7. Locally update dE and weights
    # ============================================================

    for u, v in affected:

        old_weight = weight_array[u, v]

        # local dE from cached syndrome
        touched = (
            int(b[u % H, v % L])
            + int(b[(u - 1) % H, v % L])
            + int(b[(u - 1) % H, (v - 1) % L])
        )

        new_dE = 3 - 2 * touched
        dE_array[u, v] = new_dE

        if flags[u, v]:
            new_weight = 0.0

        else:
            if weight_rule == "boltzmann":
                new_weight = np.exp(-beta * new_dE)

            elif weight_rule == "negative_boltzmann":
                if new_dE < 0:
                    new_weight = float(np.exp(-beta * new_dE))
                else:
                    new_weight = 0.0

            elif weight_rule == "metropolis":
                if new_dE <= 0:
                    new_weight = 1.0
                else:
                    new_weight = float(np.exp(-beta * new_dE))

            elif weight_rule == "bkl":
                denom = np.exp(beta * new_dE) - 1.0

                if abs(denom) < 1e-14:
                    new_weight = 1.0 / beta
                else:
                    new_weight = float(new_dE / denom)

            else:
                raise ValueError(
                    "weight_rule must be 'boltzmann', 'negative_boltzmann', 'metropolis', or 'bkl'."
                )

        weight_array[u, v] = new_weight
        total_weight += new_weight - old_weight

    total_weight = float(total_weight)

    if total_weight < 0 and abs(total_weight) < 1e-10:
        total_weight = 0.0

    # ============================================================
    # 8. Debug checks
    # ============================================================

    if debug_check:

        E_after = energy(lattice)

        if E_after != E_before + chosen_dE:
            print("Energy consistency failed.")
            print("chosen spin:", chosen_spin)
            print("chosen dE:", chosen_dE)
            print("E before:", E_before)
            print("E after:", E_after)
            print("expected:", E_before + chosen_dE)
            raise ValueError("Energy change does not match chosen dE.")

        b_full = compute_b_nm(lattice).astype(np.uint8)

        if not np.array_equal(b, b_full):
            print("Syndrome cache failed.")
            print("chosen spin:", chosen_spin)
            print("cached b:")
            print(b)
            print("full b:")
            print(b_full)
            raise ValueError("Cached syndrome b does not match compute_b_nm(lattice).")

        bb_full = b_full.astype(np.int16)

        touched_full = (
            bb_full
            + np.roll(bb_full, shift=1, axis=0)
            + np.roll(np.roll(bb_full, shift=1, axis=0), shift=1, axis=1)
        )

        dE_full = (3 - 2 * touched_full).astype(np.int8)

        if not np.array_equal(dE_array, dE_full):
            print("dE cache failed.")
            print("chosen spin:", chosen_spin)
            print("difference:")
            print(dE_array - dE_full)
            raise ValueError("Cached dE_array does not match full recomputation.")

        dE_float_full = dE_full.astype(float)

        if weight_rule == "boltzmann":
            weight_full = np.exp(-beta * dE_float_full)

        elif weight_rule == "negative_boltzmann":
            weight_full = np.where(
                dE_full < 0,
                np.exp(-beta * dE_float_full),
                0.0,
            )

        elif weight_rule == "metropolis":
            weight_full = np.where(
                dE_full <= 0,
                1.0,
                np.exp(-beta * dE_float_full),
            )

        elif weight_rule == "bkl":
            denom = np.exp(beta * dE_float_full) - 1.0
            weight_full = np.zeros_like(dE_float_full, dtype=float)

            nonzero = np.abs(denom) > 1e-14
            weight_full[nonzero] = dE_float_full[nonzero] / denom[nonzero]
            weight_full[~nonzero] = 1.0 / beta

        else:
            raise ValueError(
                "weight_rule must be 'negative_boltzmann', 'metropolis', or 'bkl'."
            )

        weight_full[flags] = 0.0

        if not np.allclose(weight_array, weight_full):
            print("weight cache failed.")
            print("max abs diff:", np.max(np.abs(weight_array - weight_full)))
            raise ValueError("Cached weight_array does not match full recomputation.")

        if not np.isclose(total_weight, weight_full.sum()):
            print("total weight failed.")
            print("cached:", total_weight)
            print("full:", weight_full.sum())
            raise ValueError("Cached total_weight does not match full recomputation.")

    return {
        "lattice": lattice,
        "b": b,
        "dE_array": dE_array,
        "weight_array": weight_array,
        "total_weight": total_weight,
        "chosen_spin": chosen_spin,
        "chosen_dE": chosen_dE,
        "beta": beta,
    }

def metropolis_cached_step(
    lattice,
    error_rate,
    b=None,
    dE_array=None,
    flags=None,
    beta=None,
    rng=None,
    debug_check=False,
):
    """
    One true Metropolis attempted update.

    Proposal:
        choose spin uniformly.

    Acceptance:
        if dE <= 0: accept
        if dE > 0: accept with probability exp(-beta*dE)

    Rejected moves leave lattice/b/dE_array unchanged.

    Uses diagonal NM geometry:
        spin (i,j) touches b[i,j], b[i-1,j], b[i-1,j-1].
    """

    import numpy as np
    import math

    if rng is None:
        rng = np.random.default_rng()

    H, L = lattice.shape

    if beta is None:
        beta = math.log((3 + error_rate) / error_rate) / 3

    if flags is None:
        flags = np.zeros((H, L), dtype=bool)
    else:
        flags = np.asarray(flags, dtype=bool)

    # ============================================================
    # 1. Initialize cached syndrome and dE array if needed
    # ============================================================

    if b is None:
        b = compute_b_nm(lattice).astype(np.uint8)
    else:
        b = b.astype(np.uint8, copy=False)

    if dE_array is None:
        bb = b.astype(np.int16)

        touched = (
            bb
            + np.roll(bb, shift=1, axis=0)
            + np.roll(np.roll(bb, shift=1, axis=0), shift=1, axis=1)
        )

        dE_array = (3 - 2 * touched).astype(np.int8)

    # ============================================================
    # 2. Uniformly propose one spin
    # ============================================================

    i = int(rng.integers(H))
    j = int(rng.integers(L))

    chosen_spin = (i, j)
    chosen_dE = int(dE_array[i, j])

    E_before = int(b.sum())

    # ============================================================
    # 3. Accept/reject
    # ============================================================

    if flags[i, j]:
        accepted = False

    elif chosen_dE <= 0:
        accepted = True

    else:
        accepted = rng.random() < np.exp(-beta * chosen_dE)

    # ============================================================
    # 4. If rejected, return unchanged cache
    # ============================================================

    if not accepted:
        return {
            "lattice": lattice,
            "b": b,
            "dE_array": dE_array,
            "chosen_spin": chosen_spin,
            "chosen_dE": chosen_dE,
            "accepted": False,
            "E_before": E_before,
            "E_after": E_before,
            "beta": beta,
        }

    # ============================================================
    # 5. Accepted: flip spin
    # ============================================================

    if debug_check:
        energy_before = energy(lattice)

    lattice[i, j] ^= 1

    # ============================================================
    # 6. Update syndrome cache locally
    # ============================================================

    toggled_stabs = [
        (i % H, j % L),
        ((i - 1) % H, j % L),
        ((i - 1) % H, (j - 1) % L),
    ]

    for a, c in toggled_stabs:
        b[a, c] ^= 1

    # ============================================================
    # 7. Update local dE cache
    # ============================================================

    affected = set()

    # If stabilizer b[a,c] changed, affected spins are:
    # (a,c), (a+1,c), (a+1,c+1)
    for a, c in toggled_stabs:
        affected.add((a % H, c % L))
        affected.add(((a + 1) % H, c % L))
        affected.add(((a + 1) % H, (c + 1) % L))

    for u, v in affected:
        touched = (
            int(b[u % H, v % L])
            + int(b[(u - 1) % H, v % L])
            + int(b[(u - 1) % H, (v - 1) % L])
        )

        dE_array[u, v] = 3 - 2 * touched

    E_after = int(b.sum())

    # ============================================================
    # 8. Optional debug checks
    # ============================================================

    if debug_check:
        energy_after = energy(lattice)

        if energy_after != energy_before + chosen_dE:
            print("Energy consistency failed.")
            print("chosen_spin:", chosen_spin)
            print("chosen_dE:", chosen_dE)
            print("energy_before:", energy_before)
            print("energy_after:", energy_after)
            print("expected:", energy_before + chosen_dE)
            raise ValueError("Metropolis dE does not match actual energy change.")

        b_full = compute_b_nm(lattice).astype(np.uint8)

        if not np.array_equal(b, b_full):
            print("Syndrome cache failed.")
            print("chosen_spin:", chosen_spin)
            print("cached b:")
            print(b)
            print("full b:")
            print(b_full)
            raise ValueError("Cached b does not match compute_b_nm(lattice).")

        bb_full = b_full.astype(np.int16)

        touched_full = (
            bb_full
            + np.roll(bb_full, shift=1, axis=0)
            + np.roll(np.roll(bb_full, shift=1, axis=0), shift=1, axis=1)
        )

        dE_full = (3 - 2 * touched_full).astype(np.int8)

        if not np.array_equal(dE_array, dE_full):
            print("dE cache failed.")
            print("chosen_spin:", chosen_spin)
            print("difference:")
            print(dE_array - dE_full)
            raise ValueError("Cached dE_array does not match full recomputation.")

    return {
        "lattice": lattice,
        "b": b,
        "dE_array": dE_array,
        "chosen_spin": chosen_spin,
        "chosen_dE": chosen_dE,
        "accepted": True,
        "E_before": E_before,
        "E_after": E_after,
        "beta": beta,
    }

    
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
    latt = copy.deepcopy(lattice)
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
    latt = copy.deepcopy(lattice)
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
    latt = copy.deepcopy(lattice)
    
    syndrome, C1 = sweeper_test(latt)
    # print("C1", C1)
    # print("syndrome", syndrome)
    E = optimal_decoder(syndrome)
    C2 = C2_constructor(E, latt)
    # print("C2", C2)
    weights = [( (R_i + C1 + C2 ) % 2 ).sum() for R_i in R]
    min_weight = min(weights)
    min_indices = [i for i, v in enumerate(weights) if v == min_weight]
    random_index = random.choice(min_indices)
    check = ((R[random_index] + C1 + C2 + lattice + R[x]) % 2 ).sum()
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
    aug = np.concatenate([copy.deepcopy(A), y.reshape(-1,1)], axis=1)

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

def correlation(spin_config: list, tau:int =4):
    spin_conf = np.array(copy.deepcopy(spin_config))
    spin_conf = 1 - 2 * spin_conf  # Convert to -1, 1
    T, H, L = spin_conf.shape
    N = H * L
    S_T = np.mean(spin_conf)
    # S_T = 1/2
    correlation = []
    correlation = []
    for t in range(T - tau):
        # For each tau, compute the correlation between t0 and t0+tau
        s0 = spin_conf[t]        # shape (H, L)
        st = spin_conf[tau]  # shape (H, L)
        # Correlator: <S(t0) S(t0+tau)> - <S>^2, averaged over sites
        # corr = np.mean(s0 * st) - S_T ** 2
        corr = np.mean((s0 * st)) - np.mean(s0) * np.mean(st) 
        correlation.append(corr)
    return T- tau, correlation


def _step_fill_uniform_1d(times, values, dt=None, n_steps=None,
                          include_start=True, include_end=False):
    """
    Resample a step function defined by (times, values) onto a uniform grid
    using left-hold (previous value) semantics.

    Parameters
    ----------
    times : 1D array-like, strictly increasing
    values: 1D array-like, same length as `times`
    dt : float, optional
        Target uniform step in physical time. If None, inferred from n_steps.
    n_steps : int, optional
        Number of uniform steps between times[0] and times[-1].
        If provided, dt=(t_end - t_start)/n_steps.
    include_start : bool, default True
        Include the initial time t0 in the output grid.
    include_end : bool, default False
        Include the final time t_end in the output grid.

    Returns
    -------
    t_out, v_out : 1D ndarray
        Uniform grid times and left-hold values.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or len(t) != len(v) or len(t) < 2:
        raise ValueError("times/values must be 1D, same length >=2")
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")

    t0, t1 = t[0], t[-1]
    span = t1 - t0
    if span <= 0:
        # Degenerate: all times equal – just return a single sample
        return (np.array([t0]) if include_start else np.array([]),
                np.array([v[0]]) if include_start else np.array([]))

    if dt is None and n_steps is None:
        raise ValueError("Specify either dt or n_steps")
    if dt is None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        dt = span / n_steps

    # Build the uniform grid
    eps = np.finfo(float).eps * max(1.0, abs(t1))
    start = t0 if include_start else (t0 + dt)
    stop  = (t1 + (eps if include_end else 0.0))
    if start > stop + 1e-15:
        # Nothing to sample
        return np.array([], dtype=float), np.array([], dtype=float)
    t_grid = np.arange(start, stop, dt)
    if include_end and (t_grid.size == 0 or t_grid[-1] < t1 - 0.5*dt):
        # Ensure exact inclusion if requested (avoid FP drift)
        t_grid = np.r_[t_grid, t1]

    # Left-hold: for each grid point τ, pick index i with t[i] <= τ < t[i+1]
    idx = np.searchsorted(t, t_grid, side='right') - 1
    # Clamp to valid range (could be -1 if τ < t0 due to numerical noise)
    idx = np.clip(idx, 0, len(t) - 1)
    v_grid = v[idx]
    return t_grid, v_grid


def step_fill_uniform(times, mags, dt=None, n_steps=None,
                      include_start=False, include_end=False, per_seed=True):
    """
    Batch wrapper. Accepts:
      - a single (times, mags) 1D pair, or
      - lists/arrays of per-realization times and mags (ragged allowed).

    If you pass multiple realizations:
      * If dt is given, that same dt is used for every realization.
      * If n_steps is given and dt is None:
          - per_seed=True  → use each seed's own (t_end - t_start)/n_steps
          - per_seed=False → compute a global dt from the *first* seed's span
                             (or you can change to use global min/max if you prefer).

    Returns
    -------
    T_out, M_out :
        If single trace → (1D t_out, 1D m_out).
        If batch        → (list of 1D arrays t_out_i, list of 1D arrays m_out_i).
    """
    # Detect single vs batch
    def _is_1d_like(x):
        return np.ndim(x) == 1

    # Single trace
    if _is_1d_like(times) and _is_1d_like(mags):
        return _step_fill_uniform_1d(times, mags, dt=dt, n_steps=n_steps,
                                     include_start=include_start, include_end=include_end)

    # Batch: each element of times/mags is a 1D array-like for one realization
    if len(times) != len(mags):
        raise ValueError("For batch input, lengths of times and mags lists must match")

    # Resolve dt per seed if needed
    if dt is not None:
        dts = [float(dt)] * len(times)
    else:
        if n_steps is None:
            raise ValueError("For batch input, specify dt or n_steps")
        if per_seed:
            dts = []
            for ti in times:
                ti = np.asarray(ti, dtype=float)
                if ti.size < 2:
                    raise ValueError("Each seed must have at least 2 timestamps")
                dts.append((ti[-1] - ti[0]) / n_steps)
        else:
            t0 = np.asarray(times[0], dtype=float)
            base_dt = (t0[-1] - t0[0]) / n_steps
            dts = [base_dt] * len(times)

    T_out, M_out = [], []
    for ti, mi, dti in zip(times, mags, dts):
        t_i, m_i = _step_fill_uniform_1d(ti, mi, dt=dti, n_steps=None,
                                         include_start=include_start, include_end=include_end)
        T_out.append(t_i)
        M_out.append(m_i)
    return T_out, M_out


def time_bin(time_lists:list, decoder_mag:list):
 def spin_classify(patterns:list, ordering = False):
    flat_each = [ [x for row in m for x in row] for m in patterns ]
    keys = []
    for flat in flat_each:
        bitstr = ''.join('1' if b else '0' for b in flat)
        keys.append(bitstr)
    classes = defaultdict(int)
    for key in keys:
        if key not in classes:
            classes[key] = 0
        classes[key] +=1 
    if ordering == True:
        indices = {}
        for key in keys:
           indices[key]= [i for i, s in enumerate(keys) if s == key]
        return classes, indices
    else:
        return classes

from itertools import product

def all_binary_matrices(n_rows: int, n_cols: int):
    """
    Yield every n_rows x n_cols matrix with entries in {0,1}, row-major order.
    Each matrix is a list of lists of ints.
    Total count: 2**(n_rows * n_cols).
    """
    for bits in product((0, 1), repeat=n_rows * n_cols):
        yield [list(bits[i*n_cols:(i+1)*n_cols]) for i in range(n_rows)]

def decstr_to_bits_width(s: str, width: int, twos_complement: bool = False) -> str:
    n = int(s.strip())
    if n >= 0:
        return format(n, f'0{width}b')[-width:]      # trim if too long
    else:
        # two's complement within given width
        return format((1 << width) + n, f'0{width}b')[-width:]




from collections import Counter
from typing import Iterable, Callable, Any, Optional

def is_subset(
    A: Iterable[Any],
    B: Iterable[Any],
    mode: str = "set",
    key: Optional[Callable[[Any], Any]] = None,
) -> bool:
    """
    Check if A is a 'subset' of B under different notions:

    mode:
      - 'set'        : ignore multiplicities and order (mathematical subset)
      - 'multiset'   : respect multiplicities (bag/multiset subset)
      - 'subsequence': order matters but elements need not be contiguous

    key:
      Optional transformer to make elements hashable/normalized
      (e.g., key=tuple for lists-of-lists; key=str for bitstrings).
    """
    if key is None:
        key = lambda x: x

    if mode == "set":
        try:
            return set(map(key, A)).issubset(set(map(key, B)))
        except TypeError:
            # fallback if elements aren’t hashable: materialize and scan
            A_norm = list(map(key, A))
            B_norm = list(map(key, B))
            return all(x in B_norm for x in A_norm)

    elif mode == "multiset":
        # counts in A must be <= counts in B for every element
        A_cnt = Counter(map(key, A))
        B_cnt = Counter(map(key, B))
        return all(A_cnt[x] <= B_cnt.get(x, 0) for x in A_cnt)

    elif mode == "subsequence":
        # each element of A must appear in B in order (not necessarily contiguously)
        it = iter(map(key, B))
        try:
            return all(next(x for x in it if x == key(a)) or True for a in A)
        except StopIteration:
            return False

    else:
        raise ValueError("mode must be 'set', 'multiset', or 'subsequence'")

import numpy as np

def time_bin(t_end, x_interval, extend_right=False):
    """
    t_end[i,k] = end time of interval k (T_{k+1})
    x_interval[i,k] = value on interval [T_k, T_{k+1})
    """
    t_end = np.asarray(t_end, float)
    x_interval = np.asarray(x_interval, float)
    nbins = np.array(t_end).shape[1]  
    if t_end.shape != x_interval.shape or t_end.ndim != 2:
        raise ValueError("t_end and x_interval must be 2D with same shape")

    t_min = 0.0
    t_max = np.nanmax(t_end)
    edges = np.linspace(t_min, t_max, nbins + 1)

    R, _ = t_end.shape
    out = np.full((R, edges.size), np.nan)

    for i in range(R):
        mask = np.isfinite(t_end[i]) & np.isfinite(x_interval[i])
        ti = t_end[i, mask]
        xi = x_interval[i, mask]
        if ti.size == 0:
            continue

        # should already be increasing, but safe:
        order = np.argsort(ti)
        ti = ti[order]
        xi = xi[order]

        # interval index k such that edge is inside [T_k, T_{k+1})
        idx = np.searchsorted(ti, edges, side="right")  # NOTE: no "-1"

        if extend_right:
            idx = np.clip(idx, 0, xi.size - 1)
            valid = (idx >= 0)
        else:
            valid = (idx >= 0) & (idx < xi.size) & (edges < ti[-1])

        out[i, valid] = xi[idx[valid]]

    mean = np.nanmean(out, axis=0)
    return edges, mean



def time_bin_1(time_lists:list, decoder_mag:list):
    """
    Bins the time lists into intervals of size bin_size.
    Returns a list of binned times and their corresponding counts.
    """
    # flatten cumsum lists in one and divide by equllength bins
    N, T = np.array(time_lists).shape
    time_finer = np.zeros((N, (T-1)*10))
    mag_finer = np.zeros((N, (T-1)*10))
    min_t = min([min(np.diff(tl)) for tl in time_lists])
    # for i in range(N):
    #     time_finer[i], mag_finer[i] = step_fill_uniform(time_lists[i], decoder_mag[i], n_steps=(T-1)*10)
    time_finer, mag_finer = step_fill_uniform(time_lists, decoder_mag, n_steps=(T-1)*30)
    flattened_time = np.concatenate(time_finer)
    flattened_mag = np.concatenate(mag_finer)

    order_time = np.argsort(np.abs(flattened_time))
    flat_sorted_by_mag_time = flattened_time[order_time]
    flat_sorted_by_mag_mag = flattened_mag[order_time]

    nbins = np.array(time_lists).shape[1]
    edges_time = np.linspace(flat_sorted_by_mag_time.min(), flat_sorted_by_mag_time.max(), nbins+1)
    centers_time = 0.5 * (edges_time[:-1] + edges_time[1:])

    edges_mag = np.linspace(flat_sorted_by_mag_mag.min(), flat_sorted_by_mag_mag.max(), nbins+1)
    centers_mag = 0.5 * (edges_mag[:-1] + edges_mag[1:])

    # create bins 
    bin_idx_1 = np.digitize(flat_sorted_by_mag_time, edges_time) - 1
    bin_idx_1 = np.clip(bin_idx_1, 0, nbins-1)

    sums_time   = np.bincount(bin_idx_1, weights=flat_sorted_by_mag_time, minlength=nbins)
    counts_time = np.bincount(bin_idx_1,           minlength=nbins)

    sums_mag   = np.bincount(bin_idx_1, weights=flat_sorted_by_mag_mag, minlength=nbins)
    counts_mag = np.bincount(bin_idx_1,           minlength=nbins)

    means_time = np.empty_like(centers_time)
    means_mag = np.empty_like(centers_mag)
    # normal division where counts>0
    np.divide(sums_time, counts_time, out=means_time, where=(counts_time>0))
    np.divide(sums_mag, counts_mag, out=means_mag, where=(counts_mag>0))
    # fill empty bins with the bin center
    valid = counts_time > 0
    empty = ~valid
    means_time[empty] = centers_time[empty]

    means_mag[empty]    = np.interp(centers_time[empty],
                                    centers_time[valid], means_mag[valid])
    return means_time,  means_mag

import multiprocessing as mp

def run_until_truncation(seed: int,
                         logical: int,
                         error_rate: float,
                         H: int,
                         L: int) -> tuple[float, list[float], list[float]]:
    random.seed(seed)
    # beta = math.log((3 + error_rate) / error_rate) / 3
    time_list = []
    energy_list = []
    mag_list =[]
    x = lattice_space(H, L)
    logicals = logical_operators(x)
    lattice = copy.deepcopy(logicals[logical])
    step = 0
    tt = {9:  91599.08858273,  
          12: 278749.52807721,
          15: 481821.1667528,
          18:   686943.27286395,
          24: 1171231.40684312,
          27: 1234491.48927047,
          30: 1098058.82823757,
          33: 796866.28682478,
          39: 769532.54229311,
          48:  566622.97092831,
          51:  456105.62961858,
          57: 407959.47402282,
          99: 275285.79732844,
          195:  254027.27736396}
    
    decoder_step = 1
    truncated = False
    while not truncated:
        step += 1
        lattice, e, t, spin_config2, total_R, spin_class_check = bkl(lattice, 1, error_rate, config = True)
        energy_list.append(e[0])
        time_list.append(t[0])
        mag_list.append(lattice.sum())
        # lattice = monte_carlo_error(lattice, 1 , error_rate)
        # if step % 10 == 0 and not logical_checks(lattice, logical):
        #     break
        if sum(time_list) > tt[H] * decoder_step / 1000:
            truncated = not logical_checks(lattice, logical)
            decoder_step += 1
            # print(f"Truncated at step {step} with total time {sum(time_list)} ms")
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
    lattice = copy.deepcopy(logicals[logical])
    # lattice = bkl(lattice, 1, [], error_rate, [])
    lattice = monte_carlo_error(lattice, 1 , error_rate)
    flag = logical_checks(lattice, logical)
    out_list.append(flag)
    


def run_until_truncation_mag(seeds: int,
                         steps: int,
                         logical: int,
                         error_rate: float,
                         H: int,
                         L: int,
                         beta1: float):
    
    E = H * L
    beta = math.log((3 + error_rate) / error_rate) / 3 if beta1 is None else beta1
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
        lattice = copy.deepcopy(logicals[logical])
        # lattice = monte_carlo_error(lattice, 1 , 0.5)
        

        energy_list = []
        time_list   = []            # if bkl populates it; otherwise you can drop it
        mags = []
        for i in range(num_steps):
            mags.append(lattice.sum())
            lattice = bkl(lattice, 1, error_rate, energy_list,  time_list)       
            if i % 20 == 0: #and not logical_checks(lattice, logical):
                mem_times.append(np.sum( time_list))
                

        if len(energy_list) != num_steps:
            raise ValueError(f"Run {run_idx}: expected {num_steps} energies, got {len(energy_list)}")

        decoder_trajs[run_idx, :] = energy_list
        decoder_times[run_idx, :] = time_list  # if bkl populates it; otherwise you can drop it
        decoder_mag[run_idx, :] = mags  # Store the magnetization trajectory
    # compute average (and normalize)
    # mean_energy = decoder_trajs.mean(axis=0) / E
    # std_decoder = decoder_trajs.std(axis=0) / E
    time_lists_full = [np.cumsum(tl) for tl in decoder_times]
    mean_times_cum, mean_energy = time_bin(time_lists_full, decoder_trajs)
    mean_times_cu, mean_mag = time_bin(time_lists_full, decoder_mag)
    # mean_times = decoder_times.mean(axis=0)  # dT list  

    # mean_mag = decoder_mag.mean(axis=0) /E  # Store the average magnetization trajectory
    mean_mem = np.mean(mem_times)
    # mean_step = np.mean(mem_times[:,0])
    
    ref_energy = 1/( 1 + np.exp( beta ))
    #=========== data save ==============
    with open('./results/mags.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # optional header
        writer.writerow(['mean_times_cum','mean_mag','mean_energy'])
        # write row by row
        writer.writerows(zip(mean_times_cum, mean_mag, mean_energy))
    # x = np.arange(5000)
    mean_energy = mean_energy / E
    mean_mag = mean_mag / E
    plt.figure(figsize=(10, 4))
    plt.plot(mean_times_cum, mean_energy, color ='blue', label='energy')
    plt.plot(mean_times_cum[:-1], mean_mag[:-1], color ='green', label='magnitization')
    plt.axvline(x=mean_mem, color='red', linestyle='--', linewidth=2, label= f'{mean_mem}')
    m = 1/2 - 1/2 * np.exp(-2*mean_times_cum/beta)
    plt.plot(mean_times_cum, m, color='orange', linestyle='--', linewidth=2, label= f'1/2 - 1/2 * exp(-t/beta)')
    plt.axhline(y = ref_energy, color='purple', linestyle='--', linewidth=2, label= f'1/(1+exp(beta))={ref_energy:.3f}')

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


def run_until_truncation_correlation(seed: int,
                                     steps: int,
                         logical: int,
                         error_rate: float,
                         H: int,
                         L: int) -> tuple[list[float], list[float]]:
    random.seed(seed)
    beta = math.log((3 + error_rate) / error_rate) / 3
    time_list = []
    energy_list = []
    mag_list =[]
    x = lattice_space(H, L)
    logicals = logical_operators(x)
    lattice = copy.deepcopy(logicals[logical])
    lattice = monte_carlo_error(lattice, 1 , 0.5)
    energy_hist = []
    time_list = []
    time = 1000
    random.seed(50)
    # lattice = monte_carlo_error(lattice, 2, error_rate)
    truncated = False
    step = 0
    error_patterns = []
    syndromes = []
    spin_config = []
    while step <steps:
        step += 1
        lattice = bkl(lattice,  1, error_rate, energy_hist, time_list)
        spin_config.append(copy.deepcopy(lattice))
        error_patterns.append((copy.deepcopy(lattice)+ copy.deepcopy(logicals[logical])) % 2) 

    _, correlations0 = correlation(spin_config, 1)
    _, correlations1 = correlation(spin_config, 4)
    _, correlations2 = correlation(spin_config, 16)
    _, correlations3 = correlation(spin_config, 56)
    
    return time_list, correlations0, correlations1, correlations2, correlations3

def main():
# def main_mag():
    parser = argparse.ArgumentParser(description='Run memory time simulation.')

    # parser.add_argument('--seeds', type=int, nargs='+', default= list(range(100)),
    #                     help='Random seed for reproducibility.')
    
    parser.add_argument('--seeds', type=int,  default=100,
                        help='Random seed for reproducibility.')
    parser.add_argument('--logical', type=int, default=3,
                        help='Index of logical operator to test (default: 3, all up).')
    
    parser.add_argument('--error_rate', type=float, default=0.0001,
                        help='Physical Error rate for the BKL update.')
    
    parser.add_argument('--beta', type=float, default=None,
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
    beta = args.beta
    run_until_truncation_mag(seeds, steps, logical, error_rate, H, L, beta1 = beta)



# def main():
def main_main():
    parser = argparse.ArgumentParser(description='Run memory time simulation.')

    parser.add_argument('--seeds', type=int,  default=2000,
                        help='Random seed for reproducibility.')
    # $(seq 0 2 1998)
    parser.add_argument('--logical', type=int, default=3,
                        help='Index of logical operator to test (default: 3, all up).')
    
    parser.add_argument('--error_rate', type=float, default=0.0001,
                        help='Physical Error rate for the BKL update.')
    
    
    parser.add_argument('--H', type=int, default=9,
                        help='Lattice Height.')

    parser.add_argument('--L', type=int, default=6,
                        help='Lattice Length.')
                        
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    logical = args.logical
    error_rate = args.error_rate
    H = args.H
    L = args.L
    print(error_rate)
    


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
        for seed, mt, en, mg in zip(seeds, mem_times, energies, mags):
            writer.writerow([
                seed,
                f"{mt:.6f}",
                json.dumps(en),
                json.dumps(mg),
            ])

    print(f"Wrote results.csv with {len(seeds)} rows.")

    mem_times = list(mem_times)

    if not mem_times:
        raise RuntimeError("No results collected—check your worker function!")
    avg_mem_times = sum( mem_times) / (len( mem_times))
    max_mem_times = max( mem_times)
    median_mem_times = np.median(mem_times)
    std_mem_times = np.std(mem_times)/np.sqrt(len(mem_times))

    time_lists_full = [np.cumsum(mem_times[k]) for k in range(len(seeds))]
    # avg_t, _, _ = time_bin_1(time_lists_full, mags)
    # print(f'Average stopping step: { max(avg_t):.2f}')
    # errors = stop_steps.count(False)/len(stop_steps)
    # print("Logical Error for each seed:", stop_steps)
    # print("logical error rate:", errors)
    print("Memory time for each seed:",  mem_times)
    print(f'Average stopping step: { avg_mem_times:.2f}')
    print("Median Mem Time:", median_mem_times)
    print("Std Error Mem Time:", std_mem_times)
    print("Max Stop:", max_mem_times)


# def main():
def main_correlation():
    parser = argparse.ArgumentParser(description='Run memory time simulation.')

    parser.add_argument('--seeds', type=int, nargs='+', default= list(range(100)),
                        help='Random seed for reproducibility.')
    
    parser.add_argument('--logical', type=int, default=3,
                        help='Index of logical operator to test (default: 3, all up).')
    
    parser.add_argument('--steps', type=int, default=13000,
                        help='steps of bkl')
    
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
    steps = args.steps
    H = args.H
    L = args.L
    
    work_args = [
        (s, steps, logical, error_rate, H, L)
        for s in seeds
    ]

    # run in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(run_until_truncation_correlation, work_args)
        # each result is (mem_time: float, energies: list[float], mags: list[float])

    # unzip the results
    time_lists, correlations0, correlations1, correlations2, correlations3 = map(list, zip(*results))

    # write CSV with JSON-encoded lists
    # with open(f'./results/correlations/results{H}*{L}*{error_rate}.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['seed', 'time_list', 'correlations_4', 'correlations_16','correlations_56',])
    #     for seed, mt, c1, c2, c3 in zip(args.seeds, time_lists, correlations1, correlations2, correlations3):
    #         writer.writerow([
    #             seed,
    #             f"{mt:.6f}",
    #             json.dumps(c1),
    #             json.dumps(c2),
    #             json.dumps(c3)
    #         ])

    # print(f"Wrote results.csv with {len(args.seeds)} rows.")

    time_lists_full = [np.cumsum(time_lists[k]) for k in range(len(seeds))]
    avg_t, _, _ = time_bin(time_lists_full)
    # avg_dt = [ sum(col)/len(col) for col in zip(*time_lists) ]
    # avg_dt = time_lists[0]  # Assuming all time lists are the same length
    avg_c0 = [ sum(col)/len(col)  for col in zip(*correlations0) ]
    avg_c1 = [ sum(col)/len(col)  for col in zip(*correlations1) ]
    avg_c2 = [ sum(col)/len(col)  for col in zip(*correlations2) ]
    avg_c3 = [ sum(col)/len(col)  for col in zip(*correlations3) ]
    
    time0 = avg_t[1:]/1
    time1 = avg_t[4:]/4
    time2 = avg_t[16:]/16
    time3 = avg_t[56:]/56
    plt.plot(time0, avg_c0, label='Correlation 1', marker='o', linestyle='-',)
    plt.plot(time1, avg_c1, label='Correlation 4', marker='o', linestyle='-',)
    plt.plot(time2, avg_c2, label='Correlation 16',marker='o', linestyle='-',)
    plt.plot(time3, avg_c3, label='Correlation 56', marker='o', linestyle='-',)
    plt.xlabel('time')
    plt.ylabel('Correlation')
    plt.xscale('log')
    plt.title('Correlation Function')
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/correlations/correlation_function{H}*{L}.png', dpi=300)
    plt.show()
    print(correlations3[0])



    
if __name__ == '__main__':
    main()
