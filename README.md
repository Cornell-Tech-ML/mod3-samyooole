# MiniTorch Module 3

## SIMPLE on GPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.1`

Epoch  0  loss  6.323657768627313 correct 41
Seconds per epoch: 1.9933
Epoch  10  loss  1.9079576748223972 correct 48
Seconds per epoch: 1.9463
Epoch  20  loss  0.3264641917727783 correct 48
Seconds per epoch: 1.8684
Epoch  30  loss  0.3960136673233314 correct 50
Seconds per epoch: 1.9524
Epoch  40  loss  0.2769489066553676 correct 50
Seconds per epoch: 1.8692
Epoch  50  loss  0.013080172690561125 correct 50
Seconds per epoch: 1.9633
Epoch  60  loss  0.03751054304516012 correct 48
Seconds per epoch: 1.8655
Epoch  70  loss  0.4200003175856277 correct 50
Seconds per epoch: 1.9429
Epoch  80  loss  0.02668707162732932 correct 50
Seconds per epoch: 1.9573
Epoch  90  loss  0.1953198799413615 correct 50

## SIMPLE on CPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.1`

Epoch  0  loss  5.337089597852492 correct 38
Seconds per epoch: 0.5354
Epoch  10  loss  1.854732494732775 correct 49
Seconds per epoch: 0.1142
Epoch  20  loss  0.8974928142920144 correct 49
Seconds per epoch: 0.1239
Epoch  30  loss  1.0268652855890774 correct 50
Seconds per epoch: 0.2219
Epoch  40  loss  0.27556743207611056 correct 50
Seconds per epoch: 0.1187
Epoch  50  loss  0.3368159556650301 correct 50
Seconds per epoch: 0.1140
Epoch  60  loss  0.36473698910849367 correct 50
Seconds per epoch: 0.1113
Epoch  70  loss  0.08409307432645065 correct 50
Seconds per epoch: 0.1112
Epoch  80  loss  0.28127984139721884 correct 50
Seconds per epoch: 0.1127
Epoch  90  loss  0.11359037930876212 correct 50

## SPLIT on GPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05`

Epoch  0  loss  11.71592169513195 correct 37
Seconds per epoch: 1.9775
Epoch  10  loss  4.238489064733035 correct 39
Seconds per epoch: 1.8452
Epoch  20  loss  2.734213941715211 correct 45
Seconds per epoch: 1.9120
Epoch  30  loss  1.709370654773785 correct 46
Seconds per epoch: 1.8245
Epoch  40  loss  1.629217336770921 correct 47
Seconds per epoch: 1.9075
Epoch  50  loss  1.77757180655412 correct 49
Seconds per epoch: 1.8244
Epoch  60  loss  0.6427508047034225 correct 47
Seconds per epoch: 1.9963
Epoch  70  loss  0.7133347372152725 correct 50
Seconds per epoch: 1.8989
Epoch  80  loss  0.5306952427662641 correct 49
Seconds per epoch: 1.8398
Epoch  90  loss  0.986783643464922 correct 50

## SPLIT on CPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`

Epoch  0  loss  5.861840182385447 correct 30
Seconds per epoch: 0.5386
Epoch  10  loss  6.4109958493419485 correct 35
Seconds per epoch: 0.1917
Epoch  20  loss  5.085506780142132 correct 41
Seconds per epoch: 0.1602
Epoch  30  loss  4.537558021276699 correct 47
Seconds per epoch: 0.1134
Epoch  40  loss  4.899905025516269 correct 44
Seconds per epoch: 0.1131
Epoch  50  loss  2.5157733867288643 correct 49
Seconds per epoch: 0.1132
Epoch  60  loss  2.778839415648227 correct 47
Seconds per epoch: 0.1145
Epoch  70  loss  2.5462193180264947 correct 48
Seconds per epoch: 0.1124
Epoch  80  loss  2.762627953810265 correct 50
Seconds per epoch: 0.1172
Epoch  90  loss  1.1744962417740878 correct 49

## XOR on GPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.08`

Epoch  0  loss  14.109272570073966 correct 29
Seconds per epoch: 1.9467
Epoch  10  loss  2.411569617639609 correct 46
Seconds per epoch: 1.9167
Epoch  20  loss  5.33430881577246 correct 46
Seconds per epoch: 1.9097
Epoch  30  loss  3.690999380183527 correct 44
Seconds per epoch: 1.8270
Epoch  40  loss  0.7535501579209922 correct 46
Seconds per epoch: 1.9059
Epoch  50  loss  1.0581432268626099 correct 48
Seconds per epoch: 1.8177
Epoch  60  loss  4.140879979351141 correct 44
Seconds per epoch: 1.9096
Epoch  70  loss  0.8908550954241061 correct 50
Seconds per epoch: 1.8328
Epoch  80  loss  1.7811612304553228 correct 50
Seconds per epoch: 1.9079
Epoch  90  loss  1.6081980181314375 correct 50
Seconds per epoch: 1.8269

## XOR on CPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.08`

Epoch  0  loss  6.807421472872758 correct 34
Seconds per epoch: 0.6575
Epoch  10  loss  4.993390824493823 correct 45
Seconds per epoch: 0.1155
Epoch  20  loss  3.1433598817329567 correct 46
Seconds per epoch: 0.1164
Epoch  30  loss  2.070870372450276 correct 49
Seconds per epoch: 0.1121
Epoch  40  loss  2.596236029346202 correct 47
Seconds per epoch: 0.1115
Epoch  50  loss  2.216699735574525 correct 47
Seconds per epoch: 0.1110
Epoch  60  loss  1.8634252563728289 correct 49
Seconds per epoch: 0.1116
Epoch  70  loss  0.8398096490331133 correct 48
Seconds per epoch: 0.1112
Epoch  80  loss  1.8089226745339153 correct 47
Seconds per epoch: 0.1133
Epoch  90  loss  1.330039331910401 correct 50
Seconds per epoch: 0.1265

## THE BIGGER MODEL: SPLIT on GPU, big (500)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 300 --DATASET split --RATE 0.05`

## `python project/parallel_check.py` output

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (163) 
-------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                                                  | 
        out: Storage,                                                                                                          | 
        out_shape: Shape,                                                                                                      | 
        out_strides: Strides,                                                                                                  | 
        in_storage: Storage,                                                                                                   | 
        in_shape: Shape,                                                                                                       | 
        in_strides: Strides,                                                                                                   | 
    ) -> None:                                                                                                                 | 
                                                                                                                               | 
        ## Optimization (3): check if tensors are stride-aligned - if so, take a direct mapping "fast path"                    | 
                                                                                                                               | 
        is_aligned = (                                                                                                         | 
            len(out_shape) == len(in_shape) and                                                                                | 
            np.array_equal(out_shape, in_shape) and                                                                            | 
            np.array_equal(out_strides, in_strides)                                                                            | 
        )                                                                                                                      | 
                                                                                                                               | 
        if is_aligned: # fast route                                                                                            | 
            for i in prange(len(out)):-----------------------------------------------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                                                     | 
        else: # slow route                                                                                                     | 
                                                                                                                               | 
                                                                                                                               | 
            for i in prange(len(out)): ## Optimization (1): parallel loops-----------------------------------------------------| #3
                ## Optimization (2): Numpy buffers: //pre-allocated// numpy arrays used to store indices during computation    | 
                out_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------------------------------------------| #0
                in_index: Index = np.zeros(MAX_DIMS, np.int32)-----------------------------------------------------------------| #1
                to_index(i, out_shape, out_index)                                                                              | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                                                      | 
                o = index_to_position(out_index, out_strides)                                                                  | 
                j = index_to_position(in_index, in_strides)                                                                    | 
                out[o] = fn(in_storage[j])                                                                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (188) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (189) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (223)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (223) 
----------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                         | 
        out: Storage,                                                                                                 | 
        out_shape: Shape,                                                                                             | 
        out_strides: Strides,                                                                                         | 
        a_storage: Storage,                                                                                           | 
        a_shape: Shape,                                                                                               | 
        a_strides: Strides,                                                                                           | 
        b_storage: Storage,                                                                                           | 
        b_shape: Shape,                                                                                               | 
        b_strides: Strides,                                                                                           | 
    ) -> None:                                                                                                        | 
                                                                                                                      | 
        is_aligned = (                                                                                                | 
        len(out_shape) == len(a_shape) == len(b_shape) and                                                            | 
        np.array_equal(out_shape, a_shape) and np.array_equal(out_shape, b_shape) and                                 | 
        np.array_equal(out_strides, a_strides) and np.array_equal(out_strides, b_strides)                             | 
        )                                                                                                             | 
                                                                                                                      | 
        if is_aligned: ## Optimization (3): avoid indexing fast path                                                  | 
            for i in prange(len(out)):--------------------------------------------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                                                               | 
        else:                                                                                                         | 
                                                                                                                      | 
                                                                                                                      | 
            for i in prange(len(out)): ## Optimization (1): main loop in parallel-------------------------------------| #8
                out_index: Index = np.zeros(MAX_DIMS, np.int32) ## Optimization (2): all indices use numpy buffers----| #4
                a_index: Index = np.zeros(MAX_DIMS, np.int32)---------------------------------------------------------| #5
                b_index: Index = np.zeros(MAX_DIMS, np.int32)---------------------------------------------------------| #6
                to_index(i, out_shape, out_index)                                                                     | 
                o = index_to_position(out_index, out_strides)                                                         | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                                               | 
                j = index_to_position(a_index, a_strides)                                                             | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                                               | 
                k = index_to_position(b_index, b_strides)                                                             | 
                out[o] = fn(a_storage[j], b_storage[k])                                                               | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (248) is hoisted out
 of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32) ## Optimization 
(2): all indices use numpy buffers
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (249) is hoisted out
 of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (250) is hoisted out
 of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (283)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (283) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        # Pre-calculate the stride for the reduction dimension     | 
        reduce_size = a_shape[reduce_dim]                          | 
        reduce_stride = a_strides[reduce_dim]                      | 
                                                                   | 
                                                                   | 
                                                                   | 
        # Optimization (1): main loop in parallel                  | 
        for i in prange(len(out)):---------------------------------| #10
            # Optimization (2): use numpy buffers for indices      | 
            out_index = np.zeros(MAX_DIMS, np.int32)---------------| #9
            to_index(i, out_shape, out_index)                      | 
            o = index_to_position(out_index, out_strides)          | 
                                                                   | 
            # Calculate base position for a_storage                | 
            base = index_to_position(out_index, a_strides)         | 
                                                                   | 
            # Initialize accumulator with first value              | 
            acc = a_storage[base]                                  | 
                                                                   | 
            # Optimization (3): inner loop with direct indexing    | 
            for s in range(1, reduce_size):                        | 
                j = base + s * reduce_stride                       | 
                acc = fn(acc, float(a_storage[j]))                 | 
                                                                   | 
            # Single write to output                               | 
            out[o] = acc                                           | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (301) is hoisted out
 of the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (322)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/samuelho/cs5781/mod3-samyooole/minitorch/fast_ops.py (322) 
---------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                           | 
    out: Storage,                                                                      | 
    out_shape: Shape,                                                                  | 
    out_strides: Strides,                                                              | 
    a_storage: Storage,                                                                | 
    a_shape: Shape,                                                                    | 
    a_strides: Strides,                                                                | 
    b_storage: Storage,                                                                | 
    b_shape: Shape,                                                                    | 
    b_strides: Strides,                                                                | 
) -> None:                                                                             | 
    """NUMBA tensor matrix multiply function.                                          | 
                                                                                       | 
    Should work for any tensor shapes that broadcast as long as                        | 
                                                                                       | 
    ```                                                                                | 
    assert a_shape[-1] == b_shape[-2]                                                  | 
    ```                                                                                | 
                                                                                       | 
    Optimizations:                                                                     | 
                                                                                       | 
    * Outer loop in parallel                                                           | 
    * No index buffers or function calls                                               | 
    * Inner loop should have no global writes, 1 multiply.                             | 
                                                                                       | 
                                                                                       | 
    Args:                                                                              | 
    ----                                                                               | 
        out (Storage): storage for `out` tensor                                        | 
        out_shape (Shape): shape for `out` tensor                                      | 
        out_strides (Strides): strides for `out` tensor                                | 
        a_storage (Storage): storage for `a` tensor                                    | 
        a_shape (Shape): shape for `a` tensor                                          | 
        a_strides (Strides): strides for `a` tensor                                    | 
        b_storage (Storage): storage for `b` tensor                                    | 
        b_shape (Shape): shape for `b` tensor                                          | 
        b_strides (Strides): strides for `b` tensor                                    | 
                                                                                       | 
    Returns:                                                                           | 
    -------                                                                            | 
        None : Fills in `out`                                                          | 
                                                                                       | 
    """                                                                                | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                             | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                             | 
                                                                                       | 
    # Parallel over first dimension                                                    | 
    for i in prange(out_shape[0]):-----------------------------------------------------| #11
        a_batch = i * a_batch_stride                                                   | 
        b_batch = i * b_batch_stride                                                   | 
                                                                                       | 
        # Middle dimension of output                                                   | 
        for j in range(out_shape[1]):                                                  | 
            # Final dimension of output                                                | 
            for k in range(out_shape[2]):                                              | 
                # Compute dot product with no global writes in inner loop              | 
                acc = 0.0                                                              | 
                                                                                       | 
                # Sum over contracting dimension                                       | 
                for l in range(a_shape[2]):                                            | 
                    acc += (                                                           | 
                        a_storage[a_batch + j * a_strides[1] + l * a_strides[2]] *     | 
                        b_storage[b_batch + k * b_strides[2] + l * b_strides[1]]       | 
                    )                                                                  | 
                                                                                       | 
                # Single global write after inner loops                                | 
                out_pos = (                                                            | 
                    i * out_strides[0] +                                               | 
                    j * out_strides[1] +                                               | 
                    k * out_strides[2]                                                 | 
                )                                                                      | 
                out[out_pos] = acc                                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

