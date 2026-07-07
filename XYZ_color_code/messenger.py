import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import numpy as np
import heapq
def as_bits01(a):
    a = np.asarray(a)
    if a.dtype.kind in "fc":
        a = (a > 0.5).astype(np.uint8)
    else:
        a = (a.astype(np.int64) & 1).astype(np.uint8)
    return a

def compute_b_nm(a):
    """
    Newman–Moore down-triangle defect (plaquette) field:
      b[i,j] = a[i,j] xor a[i+1,j] xor a[i+1,j+1]
    periodic boundaries.
    """
    a = as_bits01(a)
    a_i1_j  = np.roll(a, -1, axis=0)
    a_i1_j1 = np.roll(a_i1_j, -1, axis=1)
    return a ^ a_i1_j ^ a_i1_j1

def pascal_mod2_rows(num_rows: int, width: int | None = None) -> np.ndarray:
    """
    Build rows of (1+x)^n over GF(2).

    Output A has shape (num_rows, width), where
    A[n, k] = 1 iff coeff of x^k in (1+x)^n is 1 mod 2.
    """
    if width is None:
        width = num_rows

    A = np.zeros((num_rows, width), dtype=np.uint8)
    A[0, 0] = 1   # (1+x)^0 = 1

    for n in range(num_rows - 1):
        row = A[n]
        shifted = np.zeros_like(row)
        shifted[1:] = row[:-1]          # x * row
        A[n + 1] = row ^ shifted        # mod-2 addition

    return A

TRI6 = {
    0: (+1, +1),  # tri 1
    1: (+1, 0),   # tri 2
    2: (0, -1),   # tri 3
    3: (-1, -1),  # tri 4
    4: (-1, 0),  # tri 5    
    5: (0, +1)    # tri 6
}



def step_msgs_transient(msg, b, directions=TRI6, cap=10):
    """
    One time-step update of transient message packets.

    msg : shape (6, Lx, Ly), with -1 meaning 'no message currently here'
    b   : shape (Lx, Ly), current syndrome/defect field 0/1

    Rule:
      - messages propagate one step in their own direction and increase by 1
      - current defects emit a fresh message 1 one step downstream
      - if a defect disappears, it emits nothing new
      - old messages do NOT stay in place; they move on
      - if multiple things arrive at the same site this step, keep the minimum

    If stop_on_defect=True, a propagated message is killed when its upstream site is a defect.
    """
    b = (np.asarray(b, dtype=np.uint8) & 1).astype(bool)
    msg0 = np.asarray(msg, dtype=np.int16)

    if msg0.ndim != 3 or msg0.shape[0] != len(directions):
        raise ValueError(f"msg must have shape ({len(directions)}, Lx, Ly)")
    if b.shape != msg0.shape[1:]:
        raise ValueError(f"b shape {b.shape} must match spatial shape {msg0.shape[1:]}")

    big = np.int16(cap + 1)
    msg0 = np.where((msg0 >= 0) & (msg0 <= cap), msg0, big).astype(np.int16)

    n_dirs, Lx, Ly = msg0.shape
    out = np.full((n_dirs, Lx, Ly), big, dtype=np.int16)

    for d, shift in directions.items():
        upstream_msg = np.roll(msg0[d], shift, axis=(0, 1))
        upstream_active = np.roll(b, shift, axis=(0, 1))

        propagated = np.where(
            (upstream_msg < big) & (~upstream_active),
            np.minimum(upstream_msg + 1, big),
            big
        ).astype(np.int16)

        emitted = np.where(upstream_active, 1, big).astype(np.int16)

        out[d] = np.minimum(propagated, emitted)

    return np.where(out < big, out, -1).astype(np.int16)

def decode_triangles_parallel(msg, syndrome):
    """
    Vectorized triangle-level decoder for all triangles.

    Parameters
    ----------
    msg : ndarray, shape (6, Lx, Ly)
        Messages from directions 1..6 on every triangle.
        Use -1 for 'no message'.
    syndrome : ndarray, shape (Lx, Ly)
        Triangle syndrome bits (0 or 1).

    Returns
    -------
    out : dict containing
        min_mask   : bool array, shape (6, Lx, Ly)
            True where that direction attains the minimum on that triangle.
        up_prop    : bool array, shape (Lx, Ly)
        right_prop : bool array, shape (Lx, Ly)
        left_prop  : bool array, shape (Lx, Ly)
            Proposed triangle-local spin flips, after parity check.
        n_props    : uint8 array, shape (Lx, Ly)
            Number of distinct proposed spins before parity filtering.
        case       : uint8 array, shape (Lx, Ly)
            0 = no valid message
            1 = one-spin proposal
            2 = two-spin proposal
            3 = three-spin proposal
        accept     : bool array, shape (Lx, Ly)
            Whether parity matches syndrome.
    """
    msg = np.asarray(msg)
    syndrome = (np.asarray(syndrome, dtype=np.uint8) & 1)
    H = msg.shape[1]
    if msg.ndim != 3 or msg.shape[0] != 6:
        raise ValueError("msg must have shape (6, Lx, Ly)")
    if syndrome.shape != msg.shape[1:]:
        raise ValueError("syndrome must have shape (Lx, Ly) matching msg")

    # valid directions = message exists
    # valid = (msg >= 0) & (msg <= np.log(H))
    valid = (msg >= 0) & (msg <= int(H/2))
    has_any = valid.any(axis=0)

    # replace invalid entries by a huge number so min works
    big = np.iinfo(np.int32).max
    msg = np.asarray(msg, dtype=np.int32)
    work = np.where( valid, msg, big)

    # minimum valid message on each triangle
    m = work.min(axis=0)   # shape (Lx, Ly)
    min_mask = valid & (msg == m[None, :, :])   # shape (6, Lx, Ly)

    up_raw    = min_mask[0] | min_mask[1]   # directions 1,2
    right_raw = min_mask[2] | min_mask[3]   # directions 3,4
    left_raw  = min_mask[4] | min_mask[5]   # directions 5,6

    opposite_orientation_pair = (
            (min_mask[0] & min_mask[3]) |
            (min_mask[1] & min_mask[4]) |
            (min_mask[2] & min_mask[5])
    )
    # how many distinct spins are proposed?
    n_props = (
        up_raw.astype(np.uint8)
        + right_raw.astype(np.uint8)
        + left_raw.astype(np.uint8)
    )

    # classify your three cases
    # 0 = no valid message, 1/2/3 = number of distinct proposed spins
    case = np.where(has_any, n_props, 0).astype(np.uint8)

    # parity consistency with the syndrome
    # accept iff (# proposed spins mod 2) == syndrome
    geom_ok = (n_props != 2) | opposite_orientation_pair
    accept = (case > 0) & ((n_props & 1) == syndrome) & geom_ok
    up_prop    = up_raw & accept
    right_prop = right_raw & accept
    left_prop  = left_raw & accept

    return {
        "min_mask": min_mask,
        "up_prop": up_prop,
        "right_prop": right_prop,
        "left_prop": left_prop,
        "n_props": n_props,
        "case": case,
        "accept": accept,
    }



def power_of_two_ratio(a, b):
    a = int(a)
    b = int(b)

    if a <= 0 or b <= 0:
        return False
    if a == b:
        return True
    if np.log2(a).is_integer and np.log2(b).is_integer:
        return True
    # r = min(a, b) /(max(a, b) - min(a, b))
    # return r.is_integer()

def decode_triangles_coincidence(msg, syndrome, forbidden_mask, cap, tol=2):
    """
    Same up/right/left logic as before, but:
      - only decode on syndrome triangles
      - if the current minimum proposes a forbidden spin, try the next minimum
    """
    msg = np.asarray(msg, dtype=np.int32)
    syndrome = (np.asarray(syndrome, dtype=np.uint8) & 1)
    forbidden_mask = np.asarray(forbidden_mask, dtype=bool)

    if msg.ndim != 3 or msg.shape[0] != 6:
        raise ValueError("msg must have shape (6, Lx, Ly)")
    if syndrome.shape != msg.shape[1:]:
        raise ValueError("syndrome must have shape (Lx, Ly) matching msg")
    if forbidden_mask.shape != msg.shape[1:]:
        raise ValueError("forbidden_mask must have shape (Lx, Ly) matching msg")

    _, Lx, Ly = msg.shape
    valid = (msg >= 0) & (msg <= cap)

    chosen_mask = np.zeros((6, Lx, Ly), dtype=bool)
    up_prop = np.zeros((Lx, Ly), dtype=bool)
    right_prop = np.zeros((Lx, Ly), dtype=bool)
    left_prop = np.zeros((Lx, Ly), dtype=bool)
    n_props = np.zeros((Lx, Ly), dtype=np.uint8)
    case = np.zeros((Lx, Ly), dtype=np.uint8)
    accept = np.zeros((Lx, Ly), dtype=bool)

    for i in range(Lx):
        for j in range(Ly):
            # only allow proposals on syndrome triangles
            if syndrome[i, j] != 1:
                continue

            vals = msg[:, i, j][valid[:, i, j]]
            if vals.size == 0:
                continue

            # try candidate message levels from smallest upward
            for v in np.unique(np.sort(vals)):
                min_mask_ij = valid[:, i, j] & (msg[:, i, j] == v)
                n_min = min_mask_ij.sum()

                # message values at this triangle
                m = msg[:, i, j]
                val = valid[:, i, j]

                up_min    = bool(min_mask_ij[0] or min_mask_ij[1])   # directions 1,2
                right_min = bool(min_mask_ij[2] or min_mask_ij[3])   # directions 3,4
                left_min  = bool(min_mask_ij[4] or min_mask_ij[5])   # directions 5,6

                # [difference] require the paired messages to both exist and be close
                # up_close = (
                #     val[0] and val[1]
                #     and abs(int(m[0]) - int(m[1])) < tol
                # )

                # right_close = (
                #     val[2] and val[3]
                #     and abs(int(m[2]) - int(m[3])) < tol
                # )

                # left_close = (
                #     val[4] and val[5]
                #     and abs(int(m[4]) - int(m[5])) < tol
                # )
                # [ratio] if the ratio of messages are powers of 2
                up_close = (
                val[0] and val[1]
                and power_of_two_ratio(m[0], m[1])
                )

                right_close = (
                    val[2] and val[3]
                    and power_of_two_ratio(m[2], m[3])
                )

                left_close = (
                    val[4] and val[5]
                    and power_of_two_ratio(m[4], m[5])
                )

                # final proposal rule
                up_raw    = bool(up_min and up_close)
                right_raw = bool(right_min and right_close)
                left_raw  = bool(left_min and left_close)

                # the (1,3), (2,4), (3,5), (4,6), (6,2) pairs of message directions average.
                if n_min == 2:
                    # (1,3) -> up
                    if min_mask_ij[0] and min_mask_ij[2]:
                        up_raw = True

                    # (2,4) -> right
                    if min_mask_ij[1] and min_mask_ij[3]:
                        right_raw = True

                    # (3,5) -> right
                    if min_mask_ij[2] and min_mask_ij[4]:
                        right_raw = True

                    # (4,6) -> left
                    if min_mask_ij[3] and min_mask_ij[5]:
                        left_raw = True

                    # (6,2) -> left
                    if min_mask_ij[5] and min_mask_ij[1]:
                        left_raw = True


                n = (
                    int(up_raw)
                    + int(right_raw)
                    + int(left_raw)
                )

                c = n
                geom_ok = (n == 1) 
                ok = (c > 0) and (((n & 1) == syndrome[i, j])) and geom_ok

                if not ok:
                    continue

                # map proposed local spins to global spin coordinates
                blocked = False
                if up_raw and forbidden_mask[i, j]:
                    blocked = True
                if left_raw and forbidden_mask[(i + 1) % Lx, j]:
                    blocked = True
                if right_raw and forbidden_mask[(i + 1) % Lx, (j + 1) % Ly]:
                    blocked = True

                if blocked:
                    # try the next minimum value
                    continue

                # accept this level
                chosen_mask[:, i, j] = min_mask_ij
                up_prop[i, j] = up_raw
                right_prop[i, j] = right_raw
                left_prop[i, j] = left_raw
                n_props[i, j] = n
                case[i, j] = c
                accept[i, j] = True
                break

    return {
        "min_mask": chosen_mask,
        "up_prop": up_prop,
        "right_prop": right_prop,
        "left_prop": left_prop,
        "n_props": n_props,
        "case": case,
        "accept": accept,
    }


def proposal_source_mask_per_spin(up_prop, right_prop, left_prop):
    """
    Return source_mask[x,y], a 3-bit mask telling which neighboring
    triangle-role edges proposed spin (x,y).

    bit 0: triangle (x,y) proposed up
    bit 1: triangle (x-1,y) proposed left
    bit 2: triangle (x-1,y-1) proposed right
    """
    Lx, Ly = up_prop.shape
    mask = np.zeros((Lx, Ly), dtype=np.uint8)

    # bit 0: up proposal from triangle (x,y) lands on spin (x,y)
    mask |= up_prop.astype(np.uint8) << 0

    # bit 1: left proposal from triangle (i,j) lands on spin (i+1,j)
    # so for spin (x,y), source is left_prop[x-1,y]
    mask |= np.roll(left_prop.astype(np.uint8), shift=1, axis=0) << 1

    # bit 2: right proposal from triangle (i,j) lands on spin (i+1,j+1)
    # so for spin (x,y), source is right_prop[x-1,y-1]
    mask |= np.roll(
        np.roll(right_prop.astype(np.uint8), shift=1, axis=0),
        shift=1,
        axis=1,
    ) << 2

    return mask

def triangle_props_to_spin_union(up_prop, right_prop, left_prop):
    """
    Convert triangle-local proposed spins into a global union on spin coordinates.

    Parameters
    ----------
    up_prop, right_prop, left_prop : bool arrays, shape (Lx, Ly)
        Triangle-local proposals:
          up_prop[i,j]    means triangle (i,j) proposes its up spin
          right_prop[i,j] means triangle (i,j) proposes its right spin
          left_prop[i,j]  means triangle (i,j) proposes its left spin

    Returns
    -------
    spin_union : bool array, shape (Lx, Ly)
        spin_union[x,y] = True iff at least one triangle proposes spin (x,y).
    """
    up_prop = np.asarray(up_prop, dtype=bool)
    right_prop = np.asarray(right_prop, dtype=bool)
    left_prop = np.asarray(left_prop, dtype=bool)

    # triangle (i,j) -> global spin coordinates:
    # up    -> (i,   j)
    # left  -> (i+1, j)
    # right -> (i+1, j+1)

    up_sites = up_prop

    left_sites = np.roll(left_prop, shift=1, axis=0)

    right_sites = np.roll(
        np.roll(right_prop, shift=1, axis=0),
        shift=1, axis=1
    )

    spin_union = up_sites | left_sites | right_sites
    return spin_union


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


def init_proposal_memory(Lx, Ly, memory=5):
    return np.zeros((memory, Lx, Ly), dtype=bool)

def update_proposal_memory(history, new_flip_mask):
    history = history.copy()
    history[:-1] = history[1:]
    history[-1] = np.asarray(new_flip_mask, dtype=bool)
    return history
    
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
    
    


def decode_triangles_memory(msg, syndrome, forbidden_mask, cap):
    """
    Same up/right/left logic as before, but:
      - only decode on syndrome triangles
      - if the current minimum proposes a forbidden spin, try the next minimum
    """
    msg = np.asarray(msg, dtype=np.int32)
    syndrome = (np.asarray(syndrome, dtype=np.uint8) & 1)
    forbidden_mask = np.asarray(forbidden_mask, dtype=bool)

    if msg.ndim != 3 or msg.shape[0] != 6:
        raise ValueError("msg must have shape (6, Lx, Ly)")
    if syndrome.shape != msg.shape[1:]:
        raise ValueError("syndrome must have shape (Lx, Ly) matching msg")
    if forbidden_mask.shape != msg.shape[1:]:
        raise ValueError("forbidden_mask must have shape (Lx, Ly) matching msg")

    _, Lx, Ly = msg.shape
    valid = (msg >= 0) & (msg <= cap)

    chosen_mask = np.zeros((6, Lx, Ly), dtype=bool)
    up_prop = np.zeros((Lx, Ly), dtype=bool)
    right_prop = np.zeros((Lx, Ly), dtype=bool)
    left_prop = np.zeros((Lx, Ly), dtype=bool)
    n_props = np.zeros((Lx, Ly), dtype=np.uint8)
    case = np.zeros((Lx, Ly), dtype=np.uint8)
    accept = np.zeros((Lx, Ly), dtype=bool)

    for i in range(Lx):
        for j in range(Ly):
            # only allow proposals on syndrome triangles
            if syndrome[i, j] != 1:
                continue

            vals = msg[:, i, j][valid[:, i, j]]
            if vals.size == 0:
                continue

            # try candidate message levels from smallest upward
            for v in np.unique(np.sort(vals)):
                min_mask_ij = valid[:, i, j] & (msg[:, i, j] == v)

                # --- your original up/right/left logic ---
                up_raw    = bool(min_mask_ij[0] or min_mask_ij[1])   # directions 1,2
                right_raw = bool(min_mask_ij[2] or min_mask_ij[3])   # directions 3,4
                left_raw  = bool(min_mask_ij[4] or min_mask_ij[5])   # directions 5,6


                n = (
                    int(up_raw)
                    + int(right_raw)
                    + int(left_raw)
                )

                c = n
                geom_ok = (n == 1) 
                ok = (c > 0) and (((n & 1) == syndrome[i, j])) and geom_ok

                if not ok:
                    continue

                # map proposed local spins to global spin coordinates
                blocked = False
                if up_raw and forbidden_mask[i, j]:
                    blocked = True
                if left_raw and forbidden_mask[(i + 1) % Lx, j]:
                    blocked = True
                if right_raw and forbidden_mask[(i + 1) % Lx, (j + 1) % Ly]:
                    blocked = True

                if blocked:
                    # try the next minimum value
                    continue

                # accept this level
                chosen_mask[:, i, j] = min_mask_ij
                up_prop[i, j] = up_raw
                right_prop[i, j] = right_raw
                left_prop[i, j] = left_raw
                n_props[i, j] = n
                case[i, j] = c
                accept[i, j] = True
                break

    return {
        "min_mask": chosen_mask,
        "up_prop": up_prop,
        "right_prop": right_prop,
        "left_prop": left_prop,
        "n_props": n_props,
        "case": case,
        "accept": accept,
    }

def run_one_trial(
    error_rate,
    seed,
    H=6,
    L=3,
    steps=10,
    max_decoder_iters=1000,
):
    """
    Returns 1 if logical failure, 0 if success.
    """
    random.seed(seed)
    np.random.seed(seed)

    lattice = np.zeros((H, L), dtype=np.uint8)
    lattice = monte_carlo_error(lattice, error_rate=error_rate, num_steps=1)

    Lx, Ly = lattice.shape
    mem = 1 #Ly // 3
    spin_bin = init_proposal_memory(Lx, Ly, memory=mem)   # memory of last move only
    recent_mask = np.zeros((Lx, Ly), dtype=np.uint8)
    spin_pool = np.zeros((Lx, Ly), dtype=np.uint8)
    done = False
    counter = 0
    P_prev = np.zeros((Lx, Ly), dtype=bool)
    msg = np.full((6, Lx, Ly), -1, dtype=np.int16)

    while not done:
        counter += 1

        # rebuild transient message field from current syndrome each decoder round
        
        b = compute_b_nm(lattice)
        for _ in range(steps):
            msg = step_msgs_transient(msg, b, cap=max(1, L // 3))

        # decode using the latest syndrome field
        
        # out = decode_triangles_parallel(msg, b)
        # spin_propose = triangle_props_to_spin_union(
        #     out["up_prop"], out["right_prop"], out["left_prop"]
        # )
        # conservative
        # P_curr = spin_propose
        # spin_propose = P_prev & P_curr
        # lattice ^= spin_propose.astype(lattice.dtype)
        # P_prev = P_curr.copy()
    
        # lattice = bkl(lattice, error_rate= p_dyn, steps=1)
        
        # memory
        recent_mask = spin_bin.any(axis=0)
        
        # new_flip_mask = spin_propose & (~recent_mask.astype(bool))
        # lattice ^= new_flip_mask.astype(lattice.dtype)
        # spin_bin = update_proposal_memory(spin_bin, new_flip_mask)
        
        out = decode_triangles_coincidence(msg, b, np.zeros((Lx, Ly), dtype=bool), cap =max(1, L // 3), tol=8)
        spin_propose = triangle_props_to_spin_union(
            out["up_prop"], out["right_prop"], out["left_prop"]
        )
        new_flip_mask = spin_propose & (~recent_mask.astype(bool))
        lattice ^= new_flip_mask.astype(lattice.dtype)
        # spin_bin = update_proposal_memory(spin_bin, new_flip_mask)

    

        # recompute syndrome after applying the move
        b = compute_b_nm(lattice)

        if b.sum() == 0 or counter > max_decoder_iters:
            done = True

    # logical failure criterion: nonzero residual lattice
    # R = logical_operators(lattice)
    # weights = [( (R_i + spin_pool ) % 2 ).sum() for R_i in R]
    # min_weight = min(weights)
    # min_indices = [i for i, v in enumerate(weights) if v == min_weight]
    # random_index = random.choice(min_indices)
    # check = ((R[random_index] + spin_pool + lattice) % 2 ).sum()
    # if check == 0:
    #     return int(0)
    # else:
    #     return int(1)
    
    return int(np.sum(lattice) != 0)


def run_one_error_rate(
    error_rate,
    n_trials,
    base_seed=12345,
    H=3,
    L=6,
    steps=10,
    max_decoder_iters=1000,
    n_workers=4,
):
    """
    Run many independent trials for one error rate in parallel.
    """
    seeds = [base_seed + 100000 * int(round(1000 * error_rate)) + t for t in range(n_trials)]
    failures = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                run_one_trial,
                error_rate,
                seed,
                H,
                L,
                steps,
                max_decoder_iters,
            )
            for seed in seeds
        ]

        for fut in as_completed(futures):
            failures.append(fut.result())

    failures = np.array(failures, dtype=np.int64)
    n_fail = int(failures.sum())
    p_fail = n_fail / n_trials
    se = math.sqrt(p_fail * (1 - p_fail) / n_trials) if n_trials > 0 else float("nan")

    return {
        "error_rate": error_rate,
        "n_trials": n_trials,
        "n_fail": n_fail,
        "logical_failure_rate": p_fail,
        "stderr": se,
    }


def sweep_error_rates(
    error_rates,
    n_trials=100,
    base_seed=12345,
    H=3,
    L=6,
    steps=10,
    max_decoder_iters=1000,
    n_workers=4,
):
    """
    Sweep several error rates. Parallelism is over trials within each error rate.
    """
    results = []
    for p in error_rates:
        row = run_one_error_rate(
            error_rate=p,
            n_trials=n_trials,
            base_seed=base_seed,
            H=H,
            L=L,
            steps=steps,
            max_decoder_iters=max_decoder_iters,
            n_workers=n_workers,
        )
        print(
            f"p={p:.4f} | fail={row['n_fail']}/{row['n_trials']} "
            f"| logical failure rate={row['logical_failure_rate']:.6f} ± {row['stderr']:.6f}"
        )
        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    error_rates = np.linspace(0.005, 0.4, 40) 
    H = 15; L = 12
    df = sweep_error_rates(
        error_rates=error_rates,
        n_trials=1500,
        base_seed=1014,
        H=H,
        L=L,
        steps= 1, # H//8,
        max_decoder_iters=2000,
        n_workers=8,
    )

    print(df)
    df.to_csv(f"logical_failure_sweep_{H}_{L}.csv", index=False)
    print("Saved logical_failure_sweep.csv")