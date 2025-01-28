.. Mirage documentation master file. 
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Mirage's documentation!
====================================

Mirage is a tensor algebra superoptimizer that automatically discovers highly-optimized tensor programs for DNNs. Mirage automatically identifies and verifies sophisticated optimizations, many of which require joint optimization at the kernel, thread block, and thread levels of the GPU compute hierarchy. For an input DNN, Mirage searches the space of potential tensor programs that are functionally equivalent to the given DNN to discover highly-optimized candidates. This approach allows Mirage to find new custom kernels that outperform existing expert-designed ones.

Getting Started
---------------

- Follow the :doc:`instllation <installation>` to install Mirage
- Take a look at the :doc:`tutorials <tutorials/index>` to learn how to use Mirage to superoptimize your DNNs

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:
   
   installation
   mugraph
   cuda-transpiler
   triton-transpiler
   tutorials/index

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
