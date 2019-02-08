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

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Daniel Dantas** - *VisionGL guy and advisor professor* - [ddantas](https://github.com/ddantas)
* **Artur Santos Nascimento** - *Me, who wrote the code* - [arturxz](https://github.com/arturxz)

And a special one:
* **Stackoverflow** - *The true coder of all software on the internet* - [StackOverflow](https://stackoverflow.com/)

## License

I don't know much about licenses now. Later I'll put some here.
But you can freely use, if you put references to this code if you publish something or create a new thing using this one.

## Acknowledgments

* Software rocks!

