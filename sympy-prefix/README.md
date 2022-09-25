# SYMPY Prefix
This is a package for converting [sympy](https://www.sympy.org/en/index.html)
expressions to prefix notation.
There is standard prefix notation and a custom one with shorter sequences.

# What?
It's best explained with an example:
The sympy expression
```a + b*c```
is transformed to the list of strings
```['add', 'a', 'mul', 'b', 'c']```
This is equivalent to a depth first traversal of the tree
```mermaid
graph TD
add[add] --> a[a]
add --> mul[mul]
mul --> b
mul --> c
```

# Why?
Recently in Machine Learning there has been a seminal paper on [Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412)
where the authors show that with neural networks they can symbolically solve tasks like integration
and differential equations.
Specifically, this package was written for my Google Summer of Code (GSoC) project on [SYMBA](https://arxiv.org/abs/2206.08901).
I have been working to extend the model to use prefix notation.