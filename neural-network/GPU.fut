-- High level API which compiles to GPU code
-- It can also compile to a opencl python module (which we use: gpu.py)
-- Dependency: pyopencl
-- TOTALLY AWESOME STUFF INCOMING !!!
-- See: https://futhark-book.readthedocs.io/en/latest/interoperability.html#calling-futhark-from-python 
-- Overview: https://futhark-lang.org

-- vector * vector
let dotprod (u: []f32) (v: []f32): f32 =
  reduce (+) 0.0 (map2 (*) u v)


-- vector * matrix
entry lvecmul u b =
  map (\rowOfTrB -> dotprod u rowOfTrB)
    (transpose b)


-- matrix * matrix
-- ==
-- entry: dot
-- input { [[0.014f32, 0.056f32, -0.032f32], [0.045f32, 0.032f32, -0.067f32]] [[0.01f32], [0.01f32], [0.01f32]] }
-- output { [[0.00038f32], [0.0001f32]] }
entry dot a b =
  map(\rowOfA -> lvecmul rowOfA b) a


-- like numpy.exp
-- Calculate the exponential `e to the power of x` for all elements in the input array.
-- ==
-- entry: exp
-- input { [1f32, 2f32, 3f32] }
-- output { [2.71828183f32, 7.3890561f32, 20.08553692f32] }
entry exp (x: []f32) =
  map (\ofX -> 2.71828182 ** ofX) x


-- Negate a vector
-- ==
-- entry: negation
-- input { [1f32, 2f32, 3f32] }
-- output { [ -1.0f32, -2.0f32, -3.0f32 ] }
entry negation (x: []f32) =
  map (\ofX -> -ofX) x


-- divide (d) with vector
-- ==
-- entry: divide
-- input { 0.5f32 [1f32, 2f32, 4f32] }
-- output { [0.5f32, 0.25f32, 0.125f32] }
entry divide (d: f32) (x: []f32) =
  map (\ofX -> d / ofX) x


-- multiply (m) with a vector
-- ==
-- entry: multiply
-- input { 3.5f32 [1f32, 2f32, 5f32] }
-- output { [3.5f32, 7.0f32, 17.5f32] }
entry multiply (m: f32) (x: []f32) =
  map (\ofX -> m * ofX) x


-- multiply two vectors
-- ==
-- entry: multiply2
-- input { [1.5f32, 4f32, 5.75f32] [1.6f32, 7f32, 5.6f32] }
-- output { [2.4f32, 28f32, 32.2f32] }
entry multiply2 (x: []f32) (y:[]f32) =
  map2 (*) x y


-- multiply two matrices
-- ==
-- entry: matmultiply
-- input { [[1f32, 2f32 ,3f32], [4f32, 5f32, 6f32]] [[1f32, 2f32 ,3f32], [4f32, 5f32, 6f32]] }
-- output { [[ 1f32, 4f32, 9f32], [16f32, 25f32, 36f32]] }
entry matmultiply (x: [][]f32) (y: [][]f32) =
  map2 multiply2 x y


-- mutiply (p) with a matrix
-- entry: lmatmultiply
-- input { [[3f32, 1f32], [5f32, 7f32], [10f32, 11f32]] }
-- output { [[0.3f32, 0.1f32], [0.5f32, 0.7f32], [1.0f32, 1.1f32]] }
entry lmatmultiply (p: f32) (x: [][]f32) =
  map (multiply p) x

-- add (s) to a vector
-- ==
-- entry: add
-- input { 2.0f32 [1f32, 2f32, 3f32] }
-- output { [3.0f32, 4.0f32, 5.0f32] }
entry add (s: f32) (x: []f32) =
  map (\toX -> s + toX) x


-- add two vectors
-- ==
-- entry: add2
-- input { [1f32,2f32,1f32] [1f32,1f32,1f32] }
-- output { [2f32,3f32,2f32] }
entry add2 (x: []f32) (y: []f32) =
  map2 (+) x y


-- add two matrices
-- ==
-- entry: matadd
-- input { [[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]] [[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]] }
-- output {  [[2f32, 4f32, 6f32], [8f32, 10f32, 12f32]]}
entry matadd (x: [][]f32) (y: [][]f32) =
  map2 add2 x y


-- add (s) to a matrix
-- ==
-- entry: lmatadd
-- input { 1.75f32 [[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]] }
-- output { [[2.75f32, 3.75f32, 4.75f32], [5.75f32, 6.75f32, 7.75f32]] }
entry lmatadd (s: f32) (x: [][]f32) =
  map (\toX -> add s toX) x


-- substract (d) from a vector
-- ==
-- entry: substract
-- input { 4f32 [3f32, 5f32, 10f32] }
-- output { [1.0f32, -1f32, -6f32] }
entry substract (d: f32) (x: []f32) =
  map (\ofX -> d - ofX) x


-- substract two vectors
-- ==
-- entry: substract2
-- input { [4f32, 67f32, 24f32] [8f32, 9.4f32, 12.1f32] }
-- output { [-4f32, 57.6f32, 11.9f32] }
entry substract2 (x: []f32) (y: []f32) =
  map2 (-) x y


-- substract matrix (x) from matrix (y)
-- ==
-- entry: matsubstract
-- input { [[0.214f32],[0.001f32],[0.231f32]] [[0.51f32],[0.6f32],[0.043f32]] }
-- output { [[-0.296f32],[-0.599f32],[0.188f32]] }
entry matsubstract (x: [][]f32) (y: [][]f32) =
  map2 substract2 x y


-- substract (d) from a matrix
-- ==
-- entry: lmatsubstract
-- input { 4f32 [[3f32, 1f32], [5f32, 7f32], [10f32, 11f32]] }
-- output { [[1f32, 3f32], [-1f32, -3f32], [-6f32, -7f32]] }
entry lmatsubstract (d: f32) (x: [][]f32) =
  map (\ofX -> substract d ofX) x


-- Python example: (1 / (1 + numpy.exp(-x)))
-- The Sigmoid Function
-- ==
-- entry: sigmoid
-- input { [[0.00038f32], [0.0001f32]] }
-- output { [[0.500095f32], [0.500025f32]] }
entry sigmoid (x: [][]f32) =
  map (\ofX -> divide 1.0 (add 1.0 (exp (negation ofX)))) x
  

-- Reverse or permute the axes of an array (like numpy.transpose)
-- ==
-- entry: transp
-- input { [[1f32,2f32],[45f32,67f32]] }
-- output { [[1f32,45f32],[2f32,67f32]] }
entry transp (x: [][]f32) =
  transpose x

entry arr (x: [][]f32) =
  x