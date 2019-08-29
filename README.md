# VisionGL Python Wrapper

This is the Python Wrapper for VisionGL library.
With VisionGLlibrary, you can process images with improved performance, using your OpenCL device, and Python gives you a high-level way to pre-proccess the images.

## Getting Started

Just clone this repositore into visiongl/src/. I usually rename the directory to 'py', you can feel free to do the same.

### System Prerequisites

You need to have OpenCL drivers installed. There are a bunch of ways to do that.

On Ubuntu, you can do what this link says. Other Linux distributions are similar, but you need to find the correct commands to do what you want. 

* [ArchLinux](https://wiki.archlinux.org/index.php/GPGPU) - GPGPU on archlinux is pretty simple thing to do (Yep, something is simple in Archlinux besides broke the system when you do pacman -Syu)
* [Ubuntu 18.04 LTS](https://gist.github.com/Brainiarc7/dc80b023af5b4e0d02b33923de7ba1ed) - Ubuntu Instructions
* [Windows](https://streamhpc.com/blog/2015-03-16/how-to-install-opencl-on-windows/) - Was made on Windows 7/8 but it works on WIndows 10 as well.

### Python Prerequisites

You need some Python packages. Depending on where you are, you can install them via your package manager or pip. I'll give you the pip commands, that are universal. But if you like to manage your python packages through your package manager, do it by there! install the same package via pip and your package manager can leave you with serious headache!

```
pip install mako pyopencl scikit numpy matplotlib pil tifffile dicom
```

If you have Python 2.x and 3.x installed, you'll need to see if pip defaults to Python 3. If not, you'll need to use:


```
pip3 install mako pyopencl scikit numpy matplotlib pil tifffile dicom
```

## Running the tests

put a image called yamamoto.jpg and see the thing go.

## Built With

* [Python](https://www.python.org/) - Python 3
* [PyOpenCL](https://documen.tician.de/pyopencl/) - PyOpenCL API
* [Visual Studio Code](https://code.visualstudio.com/) - Simple IDE, have good helps to code in Python.

* [VisionGL Library](https://github.com/ddantas/visiongl) - The Library makes all happen

* [Coffe] - To get throught night
* [Tears] - Just that. 

## Versioning

I've used Github and fear if my computer breaks.

## Authors

* **Daniel Dantas** - *VisionGL guy and advisor professor* - [ddantas](https://github.com/ddantas)
* **Artur Santos Nascimento** - *Me, who wrote the code* - [arturxz](https://github.com/arturxz)

And a special one:
* **Stackoverflow** - *The true coder of all software on the internet* - [StackOverflow](https://stackoverflow.com/)

## License

Follows MIT License.

## Acknowledgments

* Software rocks!

