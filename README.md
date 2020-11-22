
# Welcome to alexlib

This package extends many native Python classes to equip you with an uneasy to tame power. Beside many minor extensions, the major classes extended are:
 
 * `list` is  extended to `List`
    * Forget that for loops exist in your life, because with this class, for loops are implicitly applied to all items.
  * `dict` is  extended to `Strcut`.
    * Combines the power of dot notation like classes and key access like dictionaries.
    
   * `pathlib` is  extended to `P`
        * `P` objects are increbily powerful for parsing paths, *no* more than one line of code is required to do **any** operation.
        
    * Some other classes that make honorable mention here are `Read` and `Save` classes. Together with `P`, they provide comprehensible support for file management. Life cannot get easier with those.

   
Furthermore, those classes are inextericably connected. Example, globbing a path `P` object returns a `List` object and so on.

You can read the details in the code to grapple the motivation and the philosophy behind its implementation mechanics. Fill your life with one-liners, take your code to artistic level of brevity and readability while simulataneously being more productive.


# Install
just do this in your command line
`pip install alexlib`

# Getting Started
That's as easy as taking candy from a baby; whenever you start a Python file, preface it with this

```
import alexlib.toolbox as tb
```
This unleashes the library.


# A Taste of Power
Suppose you want to know how many lines of code in your repository. The procedure is to glob all `.py` files recursively, read string code, split it by lines, count the the lines, add up everything.


To achieve this, all you need is a imminently readable one-liner.
```
tb.P.cwd().myglob("*.py, r=True).read_text().split('\n').apply(len).np.sum()
```

How does this make perfect sense?
* `myglob` returns `List` of `P` path objects
* `read_text` is a `P` method, but it is being run agaist `List` object. Behind the scense **responsible black magic** fails to find such a method in `List` and realizes it is a method of items inside the list, so it reads all files and containerize them in another `List` object and return it.
* Similar story applies to the next methods. 
* `.np` converts `List` to `numpy` array, then `.sum` is a method of `numpy`, which gives the final result.

This is the power of implicit for loops. Share with us your one-liner snippets to add it to use-cases of this package.

