
<p align="center">

<a href="https://github.com/thisismygitrepo/crocodile/commits">
<img src="https://img.shields.io/github/commit-activity/m/thisismygitrepo/crocodile" />
</a>

</p>

# Welcome to crocodile

Pythonically facilitate laborious `file management`, `distributed computing`, `scripting` and  `deep learning` workflows.

### Demo for distributing jobs on local machines:

https://user-images.githubusercontent.com/46288868/213916407-bf6300bd-3409-4a20-b71c-00374f875c7a.mp4

With this cluster management tool, old laptops and desktops are endowed second lives as cluster workhorses to which you offset heavy computations from your little personal laptop.


##### Other Distriubted Computing Features:
* 🧑‍💻 Repository containing the executed function is automatically zipped and copied to each remote.
* 📁 Binary data is `sftp`'d automatically to each remote.
* 💽💻💿 Specs of each resource is inspected and workload is distributed accordingly on the cluster.
* 📨📩 Email Notifications. Get **optionally** notified about start and finish of your submitted jobs.
* 🔒🔑 Resources locking. A Job can *optionally* hold the resources to itself and other submitted jobs will have to wait. 
  * 🙋‍♂️🙋‍♀ ️This feature enable sending aribtrary number of jobs in one go and never worry about overwhelming the remote. Then you come later and get all results.
* Zellij session with reasonable layout is fired automatically on each remote.


#### Demo for Croshell
Croshell aims at facilitating the use of Python in scripting, thus, offering an alternative to `PowerShell` & `Bash` which have absurdly complex commands that are nothing but jumble of ad-hoc developments piled over decades to save some programmers a key stroke or two. This heritage poses huge burden on the people coming into the computer science field. A full rant bashing those shells by `Brian Will` [is here](<https://www.youtube.com/watch?v=L9v4Mg8wi4U`>).

The core rationale is:
* No one has the time to listen to hours long tutorials on how powerful and versatile `ls` or `grep` are, let alone keeping the random syntax in mind (unless used on daily basis).
* Python shell on the other hand, offers benign syntax and eminent readibility but it comes at the rather hefty cost of terseness, or the lack of it. For example, to make up for just `ls`, you need to import some libraries and it will eventually set you back a couple of lines of code. That's not acceptable for the simple task of listing directory contents, let alone a task of compressing a directory.
* Crocodile comes here to make Python terser and friendlier by offering functionality for everyday use, like **file management, SSH, environment variables management, etc**. In essence, croshell to IPython is what IPython to Python shell is; that is, the basic Python shell that can only do arithmetic is turbo-boosted making it perfect for everyday errands.
* The library, if used in coding, will fill your life with one-liners, take your code to artistic level of brevity and readability while simultaneously being more productive by typing less boilerplate lines of code that are needless to say.

The name `crocodile` signifies the use of brute force in its implementation. The focus is on ease of use, as oppoesd to beating the existing shells in speed.
Mind you, speed is not an issue in 99% of everyday chores.
`Crocodile` designed carefully to be loved, learning curve cound't be flattened further.

This package extends many native Python classes to equip you with an uneasy-to-tame power. The major classes extended are:

 * `pathlib.Path` is  extended to `P`
      * Forget about importing all the **archaic** Python libraries `os`, `glob`, `shutil`, `sys`, `zipfile` etc. `P` makes the path an object, not a lame string. `P` objects are incredibly powerful for parsing paths, *no* more than one line of code is required to do **any** operation. Take a squint at this one line file wrangler:
        * get a temporary file name
        * writes `lol` text to it
        * copy it to same location (with a suffix like `_copy1`)
        * moves it to parent directory
        * converts user home to `~`
        * zip it
        * delete it
        * touch it
        * go to its parent
        * search for all files in it and select the first one.
        * upload it to the cloud (transfer.sh)
        * open the browser with the url
        * download it (by default it goes to `~/Downloads`)
        * encrypt it with a password.
        * create a symlink to it from `~/toy`
        * resolve the symbolic link
        * calculate the checksum of the file
 
```python
P.tmpfile().write_text("lol").copy().move("..", rel2it=True).collapseuser().zip().delete(sure=True).touch().parent.search("*", folders=False)[0].share_on_cloud()().download().encrypt(pwd="haha").symlink_from("~/toy").resolve().checksum()
```

```python
path = P("dataset/type1/meta/images/file3.ext")
>> path[0]  # allows indexing! makes sense, hah?
 P("dataset")
>> path[-1]  # nifty!
 P("file3.ext")
>> path[2:-1]  # even slicing!
 P("meta/images/file3.ext")
```
 * `list` is  extended to `List`
   * Forget that `for` loops exist, because with this class, `for` loops are implicitly used to apply a function to all items.
     Inevitably while programming, one will encounter objects of the same type and you will be struggling to get a tough grab on them.  `List` is a powerful structure that put at your disposal a grip, so tough, that the objects you have at hand start behaving like one object. Behaviour is ala-JavaScript implementation of ``forEach`` method of Arrays.

 * `dict` is  extended to `Struct`.
     * Combines the power of dot notation like classes and key access like dictionaries.

 * Additionally, the package provides many other new classes, e.g. `Read` and `Save`. Together with `P`, they provide comprehensive support for file management. Life cannot get easier with those. Every class inherits attributes that allow saving and loading in one line.

   
Furthermore, those classes are inextricably connected. For example, globbing a path `P` object returns a `List` object. You can move back and forth between `List` and `Struct` and `DataFrame` with one method, and so on.

* Deep Learning Modules.
  * A paradigm that facilitates working with deep learning models that is based on a tri-partite scheme:
    * HyperParameters: facilitated through `HParams` class.
    * Data: facilitated though `DataReader` class.
    * `BaseModel` is a frontend for both `TensorFlow` & `Pytorch` backends. The wrapper worked in tandem.
  * The aforementioned classes cooperate together to offer sealmess workflow during creation, training, and saving models.


# Install
In the commandline:
`pip install crocodile`.

Being a thin extension on top of almost pure Python, you need to worry **not** about your venv, the package is not aggressive in requirements, it installs itself peacefully, never interfere with your other packages. If you do not have `numpy`, `matplotlib` and `pandas`, it simply throws `ImportError` at runtime, that's it.

[comment]: # (The package is not fussy about versions either. It can though at runtime, install packages on the fly, e.g. `dill` and `tqdm` which are very lightweight libraries.)

# Install Croshell Terminal
For `Windows` machines, run the following in elevated `PowerShell`:
`Warning: This includes dotfiles manager that you might not want.`
```ps1
Invoke-WebRequest https://raw.githubusercontent.com/thisismygitrepo/machineconfig/main/src/machineconfig/setup_windows/croshell.ps1 | Invoke-Expression
```

# Getting Started
That's as easy as taking candy from a baby; whenever you start a Python file, preface it with following in order to unleash the library:

```
import crocodile.toolbox as tb
```


# A Taste of Power
EX1: Get a list of `.exe` available in terminal.

```python
     P.get_env().PATH.search('*.exe').reduce(lambda x, y: x+y).print()
```

EX2: Suppose you want to know how many lines of code in your repository. The procedure is to glob all `.py` files recursively, read string code, split each one of them by lines, count the lines, add up everything from all strings of code.


To achieve this, all you need is an eminently readable one-liner.
```python
tb.P.cwd().search("*.py", r=True).read_text().split('\n').apply(len).to_numpy().sum()
```

How does this make perfect sense?
* `search` returns `List` of `P` path objects
* `read_text` is a `P` method, but it is being run against `List` object. Behind the scenes, **responsible black magic** fails to find such a method in `List` and realizes it is a method of items inside the list, so it runs it against them and thus read all files and containerize them in another `List` object and returns it.
* A similar story applies to `split` which is a method of strings in Python.
* Next, `apply` is a method of `List`. Sure enough, it lives up to its apt name and applies the passed function `len` to all items in the list and returns another `List` object that contains the results.
* `.to_numpy()` converts `List` to `numpy` array, then `.sum` is a method of `numpy`, which gives the final result.

Methods naming convention like `apply` and `to_numpy` are inspired from the popular `pandas` library, resulting in almost non-existing learning curve.

# Friendly interactive tutorial.
Please refer to [Here](<https://github.com/thisismygitrepo/crocodile/blob/master/tutorial.ipynb>) on the main git repo.

# Full docs:
Click [Here](<https://crocodile.readthedocs.io/en/latest/>)

# Author
Alex Al-Saffar. [email](mailto:programmer@usa.com)
