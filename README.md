# VENT - Vulkan Efficient Numerical Toolkit

VENT is a high-level numerical toolkit designed for Vulkan. It is written in C++ and provides an efficient and easy-to-use interface for Vulkan in C++ projects.

## Features

- High-level interface for Vulkan
- Efficient numerical computations
- Easy to use in C++ projects

## Adding VENT to your project

To use vent in your project, you need to include the vent.h header file in your source code. 

You also need to have in your system:

- Vulkan SDK
- OpenMP



## Usage

VENT uses a syntax similar to the standard C++ library, for functions `vent::transform`, `vent::transform_reduce` and `vent::reduce`,
replacing the lambdas for transform with strings that represent functions in GLSL and limiting having a limited set of reduce operations.

You can also pass extra data in the functions by passing a tuple of pairs with the data and the name of the variable in the GLSL code.

### `vent::transform`
```cpp
// Example of a simple transform operation
std::vector<int> input = {1, 2, 3, 4, 5};
std::vector<float> output(input.size());

int m = 3;
vent::transform(
        input.begin(), // Input iterators
        input.end(), 
        output.begin(), // Output iterator
        "float res(int val) {return float(multiplier*val*val);}", // transform function
        std::make_tuple(std::make_pair("multiplier", m)) // additional variables
);
// output = {3, 12, 27, 48, 75}

```

Transform also supports binary operations

```cpp
// Example of a simple transform operation
std::vector<int> input1 = {1, 2, 3, 4, 5};
std::vector<int> input2 = {5, 4, 3, 2, 1};
std::vector<float> output(input1.size());

vent::transform(
        input1.begin(), // Input iterators
        input1.end(), 
        input2.begin(),
        output.begin(), // Output iterator
        "float res(int val1, int val2) {return float(val1*val2);}" // transform function
);
// output = {5, 8, 9, 8, 5}

```

### `vent::transform_reduce` and `vent::reduce`

```cpp
    // Example of a simple transform_reduce operation
std::vector<int> input = {1, 2, 3, 4, 5};
int m = 3;
int r0 = vent::reduce(
    input.begin(), // Input iterators
    input.end(),
    10, // initial value
    vent::ReduceOperation::add // reduce operation
); // r0 = 25

int r1 = vent::transform_reduce(
    input.begin(), // Input iterators
    input.end(),
    0, // initial value
    vent::ReduceOperation::add, // reduce operation
    "int res(int val) {return multiplier*val;}", // transform function
    std::make_tuple(std::make_pair("multiplier", m)) // additional variables
); // r1 = 45

// binary transform reduce is dot product by default
int r2 = vent::transform_reduce(
    input.begin(), // Input iterators
    input.end(),
    input.begin(), // Second iterator
    0 // initial value
); // r2 = 55
```

### `vent::linsolve`

The linsolve function can use either jacobi or Conjugate Gradient to solve a 
system o linear equations. The choice depends on if you specify if the matrix is positive definite or not.

```cpp
    uint32_t size = 2;
std::vector<float> matrix = {
        4, 1,
        1, 3
};
std::vector<float> vector = {1, 2};

std::vector<float> resultsCG(size);
std::vector<float> resultsJCB(size);

vent::linsolve(
        matrix.begin(), // matrix iterators
        matrix.end(), 
        vector.begin(), // vector iterators
        resultsCG.begin(), // output iterator
        200, // max iterations
        1e-4, // tolerance
        true // positive definite
); // resultsCG = {0.0909091, 0.636364}

vent::linsolve(
        matrix.begin(), // matrix iterators
        matrix.end(),
        vector.begin(), // vector iterators
        resultsJCB.begin(), // output iterator
        200, // max iterations
        1e-4, // tolerance
        false // positive definite
); // resultsJCB = {0.0909087, 0.636361}
```


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.