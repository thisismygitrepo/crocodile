
# Welcome to alexlib

Fill your life with one-liners, take your code to artistic level of brevity and readability while simultaneously being more productive by typing less boilerplate lines of code that are needless to say.

This package extends many native Python classes to equip you with an uneasy-to-tame power. The major classes extended are:
 
 * `list` is  extended to `List`
    * Forget that `for` loops exist in your life, because with this class, `for` loops are implicitly applied to all items.
      Inevitably while programming, one will encounter objects of the same type and you will be struggling to get a tough grab on them.  `List` is a powerful structure that put at your disposal a grip, so tough, that the objects you have at hand start behaving like one object. This is the power of implicit `for` loops.

  * `dict` is  extended to `Struct`.
    * Combines the power of dot notation like classes and key access like dictionaries.
    
   * `pathlib.Path` is  extended to `P`
        * `P` objects are incredibly powerful for parsing paths, *no* more than one line of code is required to do **any** operation. Take a shufti at this:
        ```
     path = tb.P("dataset/type1/meta/images/file3.ext")
     >> path[0]  # ALLOWS INDEXING !
        P("dataset")
     >> path[-1]  # NIFTY
        P("file3.ext")
     >> path[2:-1]  # Even slicing!
        P("meta/images/file3.ext")
     ```
     This and much much more, is only on top of the indesipensible `pathlib.Path` functionalities.
        
   * Some other classes that make honorable mention here are `Read` and `Save` classes. Together with `P`, they provide comprehensible support for file management. Life cannot get easier with those. Every class inherits attributes that allow saving and reloading in one line.

   
Furthermore, those classes are inextricably connected. Example, globbing a path `P` object returns a `List` object. You can move back and forth between `List` and `Struct` with one method, and so on.


# Install
In the commandline:
`pip install alexlib`.

Being a thin extension on top of almost pure Python, you need to worry **not** about your venv, the package is not aggressive in requirements, it installs itself peacefully, never interfere with your other packages. If you do not have `numpy`, `matplotlib` and `pandas`, it simply throws `ImportError` at runtime, that's it. The package is not fussy about versions either. It can though at runtime, install packages on the fly, e.g. `dill` and `tqdm` which are very lightweight libraries.

# Getting Started
That's as easy as taking candy from a baby; whenever you start a Python file, preface it with following in order to unleash the library:

```
import alexlib.toolbox as tb
```


# A Taste of Power
Suppose you want to know how many lines of code in your repository. The procedure is to glob all `.py` files recursively, read string code, split each one of them by lines, count the lines, add up everything from all strings of code.


To achieve this, all you need is an eminently readable one-liner.
```
tb.P.cwd().myglob("*.py", r=True).read_text().split('\n').apply(len).to_numpy().sum()
```

How does this make perfect sense?
* `myglob` returns `List` of `P` path objects
* `read_text` is a `P` method, but it is being run against `List` object. Behind the scenes, **responsible black magic** fails to find such a method in `List` and realizes it is a method of items inside the list, so it runs it against them and thus read all files and containerize them in another `List` object and returns it.
* Similar story applies to `split` which is a method of strings in Python.
* Next, `apply` is a method of `List`. Sure enough, it lives up to its apt name and applies the passed function `len` to all items in the list and returns another `List` object that contains the results.
* `.to_numpy()` converts `List` to `numpy` array, then `.sum` is a method of `numpy`, which gives the final result.


# Full docs:
Click [Here](<https://alexlib.readthedocs.io/en/latest/>)
