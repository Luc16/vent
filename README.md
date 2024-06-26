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

// Transform also supports binary operations
std::vector<int> input2 = {1, 2, 3, 4, 5};

vent::transform(
        input.begin(), // Input iterators
        input.end(), 
        input2.begin(), // Input2 iterators
        output.begin(), // Output iterator
        "float res(int val, int val2) {return float(val*val2);}", // transform function
);

```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.