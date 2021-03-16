# Low-rank registration (lowrr)

Low-rank registration of slightly misaligned images for photometric stereo.
This repository holds both the `lowrr` library, and a `lowrr` command-line executable.

> Matthieu Pizenberg, Yvain Quéau, Abderrahim Elmoataz,
> "Low-rank registration of images captured under unknown, varying lighting".
> International Conference on Scale Space and Variational Methods in Computer Vision (SSVM).
> 2021.

The algorithm presented here takes advantage of the fact that well aligned sets of images
should form a matrix with low rank.
We thus minimize the nuclear norm of that matrix (sum of singular values),
which is the convex relaxation of its rank.

This algorithm gives convincing results in the context of photometric stereo images,
which is where we have evaluated it,
but it should also work reliably in other situations where minimizing the rank makes sense.
Some additional experiments show interesting results with multimodal images for example.

![Alignment of photometric stereo images improves the 3D reconstruction][handheld]

The previous figure showcases the improvement on both the 3D reconstruction,
and the recovered albedo after an alignment of handheld photometric stereo images
of the Bayeux Tapestry.

[handheld]: https://mpizenberg.github.io/resources/lowrr/handheld.jpg

## Acknowledgements

This work was supported by the RIN project "Guide Muséal",
and by the ANR grant "Inclusive Museum Guide" (ANR-20-CE38-0007).
The authors would like to thank C. Berthelot at the Bayeux Tapestry Museum
for supervising the image acquisition campaign of the Bayeux Tapestry.

## Installation

To install the `lowrr` command-line program,
simply download the archive for your platform (Windows, MacOS, Linux)
from the [latest release][releases].
Then extract it and put the executable in a directory listed in your `PATH` environment variable.
This way, you will be able to call `lowrr` from anywhere.

[releases]: https://github.com/mpizenberg/lowrr/releases

## Usage

The simplest way to use `lowrr` is to call it with a glob pattern
for the images you want to align, for example:

```sh
lowrr img/*.png
```

By default, this will compute the registration and output to stdout
the affine parameters of each image transformation as specified
in our research paper.

If you also want to apply the transformation and save the registered images,
you can add the `--save-imgs` command line argument.

```sh
# Apply the transformation and save the registered images
lowrr --save-imgs img/*.png
```

Usually, the algorithm can estimate the aligning transformation without working
on the whole image, but just a cropped area of the image to make things faster.
You can specify that working frame with the command line arguments
`--crop <left> <top> <right> <bottom>` where the border coordinates of that frame
are specified after the `--crop` argument (top-left corner is 0,0).
In that case, I'd suggest to also add the `--save-crop` argument
to be able to visualize the cropped area and its registration.

```sh
# Work on a reduced 500x300 cropped area and visualize its registration
lowrr --crop 0 0 500 300 --save-crop img/*.png
```

You can also customize all the algorithm parameters.
For more info, have a look at the program help.

```sh
# Display the program help for more info
lowrr --help
```

## Step by step example usage

1. Install `lowrr` as described in the `Installation` section above.
   Make sure it is available to the command line by running `lowrr --help`,
   which should display the help menu of the program.
2. Download and extract this [example set of 6 photos][bd-zip]
   of the cover of a comic book about the city of Caen.
3. Open a terminal in the directory containing those images and run

```sh
lowrr --crop 2300 2300 2800 2800 --save-crop --save-imgs -v *.jpg > params.txt
```

This command will load the 6 images into memory
and extract a working area corresponding to the 500x500 frame
located between left, top, right, bottom coordinates of
2300, 2300, 2800 and 2800 respectively.
It will then perform the registration algorithm on that working area
with default parameters and save the registered images for that frame.
Finally it will project the computed registration parameters from the cropped area
to the frame of the whole image and output those into the file `params.txt`.
Each line of `params.txt` contains the affine parameters `p1, p2, p3, p4, p5, p6`
of the corresponding image, such that they form the following affine matrix:

```txt
| 1 + p1,     p3, p5 |
|     p2, 1 + p4, p6 |
|      0,      0,  1 |
```

In addition this also apply the computed registration to all images and save them on disk.
All saved images will be located in the `out/` directory.

[bd-zip]: https://unicloud.unicaen.fr/index.php/s/tBjo2YtwXHBqe7j/download

## Lib documentation

In addition to the `lowrr` executable compiled from `src/main.rs`,
we also provide the code in the form of a library,
so that it can easily be re-used for other Rust applications.
The API documentation of the library is available at
TODO: change that
https://matthieu.pizenberg.pages.unicaen.fr/low-rank-registration

## Unfamiliar with Rust?

If you want to read the source code but are not very familiar
with the Rust language, here are few syntax explanations.

Basically, if you know how to read C/C++ code, the structure of Rust
code should be pretty familiar.
For example, it uses curly braces to delimit code blocks
and the parts between brackets `<T>` are type parameters,
like templates in C++.

Here are code examples of some patterns and syntax that may be new though.

```rust
// Pattern 1: closures
let square = |x| x * x;
square(3) // -> 9

// Pattern 2: iterators
xCollection.iter().map(|x| f(x)).collect();

// Pattern 3: zipping iterators
xCollection.iter()
    .zip(yCollection.iter())
    .map(|(x,y)| f(x,y)).collect();

// Pattern 4: for loops on iterators
for x in xCollection.iter() {
    do_something_with(x)
}

// Pattern 5: crashing on potential errors
result.unwrap();
// or
result.expect("crash with an error message");
```

The first pattern is the usage of "closures",
a.k.a. "anonymous functions", a.k.a. "lambda functions".
The part between the bars `|x|` are the arguments.
The part after the bars `x * x` is the returned value.
Closures are useful to use instead of defining properly
named functions in some parts of the program.

The second pattern (`.iter().map(...)`) is basically saying that
we are iterating over a collection of things and we apply
the same function `f` to all those elements of the collection.
The `collect()` at the end is more or less saying that we are done
modifying it in this iterator, and we can regenerate a new
data structure that will contain the result of those modifications.

The third pattern consists in using `iterator1.zip(iterator2)`.
It is just to bring together two iterators and apply a function
to both elements at the same time.

Pattern 4 is another way of iterating, similar to pattern 1.
Depending on the situation, using loops or mapping a function will be more appropiate.

Finally, the usage of `unwrap()` or `expect(...)` is just to say
to the compiler that I know it is safe to extract a potentially failing value
even though it may result in an error.
In the case of an error, this will crash the program,
and print the message inside the `expect(...)`.

## Code contribution

To compile the source code yourself, you just need to install [Rust][rust],
and then run the command `cargo build --release` at the root of this project.
Cargo is Rust build tool, it will automatically download dependencies
and compile all the code.
The resulting binary will be located in `target/release/`.
The first compilation may take a little while, but then will be pretty fast.

[rust]: https://www.rust-lang.org/tools/install

## Reproducing the paper figures

> Warning: this has not been tested on Windows and Mac, only Linux.

Some figure need to run a photometric stereo reconstruction
and are not reproducible directly with the code in this directory
since that is out of scope.
All the figures that do not involve 3D reconstruction though
are reproducible with the code provided in this repository.
Most of them need to be able to run Matlab code, I leave that to you.

First, you need to build the main `lowrr` executable.

```sh
cargo build --release
```

Then, you need to build the `warp_crop` example program.

```sh
cargo build --release --example warp_crop
```

Both these executable will be located at `target/release/lowrr`
and `target/release/examples/warp_crop` respectively.
Copy them somewhere in your path to have them available
when we run the matlab scripts.

Now you need to [download the DiLiGent dataset][diligent] and extract it.
We are interested in the `pmsData/` directory containing
photometric stereo images of 10 objects.

The `eval/` directory contains two scripts `eval_registration.m`
and `eval_all_displacement_errors.m`,
as well as some helper functions for each of those scripts.
Once the DiLiGent data has been downloaded and extracted,
Change the path of `diligent_dir` in `eval_registration.m`,
eventually also lower the `nb_random` to something small,
and run the `eval_registration` Matlab script.

Once the registration is done for each algorithm on all sequences,
change `nb_random` in `eval_all_displacement_errors.m` to match
what you set in the first script and run the `eval_all_displacement_errors` matlab script.
This will generate all the visualizations available in the paper and others.

[diligent]: https://drive.google.com/uc?id=1EgC3x8daOWL4uQmc6c4nXVe4mdAMJVfg&export=download
