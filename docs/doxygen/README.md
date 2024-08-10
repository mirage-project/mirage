# Doxygen Documentation

This directory holds the configuration file for building
the HTML Doxygen documentation for the C++ and Python code.
This documentation is mainly for the developers of FlexFlow for now.

## Generate documentation locally

1. Install [doxygen](https://www.doxygen.nl/index.html). The configuration file is based on Doxygen 1.9.3. But all recent Doxygen versions should work.
2. Define `$FF_HOME` environmental variable to be the root directory of the FlexFlow repo.
3. Run Doxygen with `doxygen $FF_HOME/docs/doxygen/Doxyfile`
4. Now, you may browse the docs by opening the index page `$FF_HOME/docs/doxygen/output/html/index.html` with your favorite web browser.
