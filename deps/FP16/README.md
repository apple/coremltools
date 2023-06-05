# FP16
Header-only library for conversion to/from half-precision floating point formats

## Features

- Supports IEEE and ARM alternative half-precision floating-point format
    - Property converts infinities and NaNs
    - Properly converts denormal numbers, even on systems without denormal support
- Header-only library, no installation or build required
- Compatible with C99 and C++11
- Fully covered with unit tests and microbenchmarks

## Acknowledgements

[![HPC Garage logo](https://github.com/Maratyszcza/PeachPy/blob/master/logo/hpcgarage.png)](http://hpcgarage.org)
[![Georgia Tech College of Computing logo](https://github.com/Maratyszcza/PeachPy/blob/master/logo/college-of-computing.gif)](http://www.cse.gatech.edu/)

The library is developed by [Marat Dukhan](http://www.maratdukhan.com) of Georgia Tech. FP16 is a research project at [Richard Vuduc](http://vuduc.org)'s HPC Garage lab in the Georgia Institute of Technology, College of Computing, School of Computational Science and Engineering.

This material is based upon work supported by the U.S. National Science Foundation (NSF) Award Number 1339745. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of NSF.
