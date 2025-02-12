
"""plots
"""

import pandas as pd
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event
from matplotlib import widgets
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors  # type: ignore # noqa # pylint: disable=unused-import
from matplotlib.colors import CSS4_COLORS  # type: ignore # pylint: disable=unused-import
from matplotlib.image import AxesImage
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from crocodile.core import timestamp, Save, validate_name
from crocodile.file_management import P, OPLike, PLike
from crocodile.meta import Terminal
from crocodile.msc.odds import Cycle

import enum
import subprocess
import platform
from typing import Any, Optional, Literal, TypeAlias, Callable, Type, Union


STREAM: TypeAlias = Literal['clear', 'accumulate', 'update']  # Streaming (Refresh mechanism): * Clear the axis. (slowest, but easy on memory) * accumulate, using visibility to hide previous axes. (Fastest but memory intensive)  * The artist has an update method. (best)
ARTIST: TypeAlias = Literal['internal', 'external']  # How is the data visualized? You need artist. The artist can either be internal, as in ImShow or passed externally (especially non image data)
PARSER: TypeAlias = Literal['internal', 'external']  # Data parsing: internal for loop to go through all the dataset passed. # Allows manual control over parsing. external for loop. It should have add method. # Manual control only takes place after the external loop is over.
SAVE_TYPE: TypeAlias = Literal['Null', 'PDF', 'PNG', 'MPEGPipeBased', 'MPEGFileBased', 'GIFFileBased', 'GIFPipeBased']
PLT_CMAPS: list[str] = plt.colormaps()  # type: ignore


class FigurePolicy(enum.Enum):
    close_create_new = 'Close the previous figure that has the same figname and create a new fresh one'
    add_new = 'Create a new figure with same path but with added suffix'
    same = 'Grab the figure of the same path'


def assert_requirements():
    try: subprocess.check_output(['where.exe' if platform.system() == 'Windows' else 'which', 'magick'])
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        # P(r"https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z").download().unzip().search()[0].rename("ffmpeg").move(r"C://")  # P("C:\\ffmpeg\\bin")  # add to PATH
        print("Installinng image magick")
        if platform.system() == "Windows":
            Terminal().run("winget install ImageMagick.ImageMagick", shell="powershell")  # gives ffmpeg as well
            print("You might need to restart your machine before PATH change impact takes place.")
        else: raise NotImplementedError from err


class GenericSave:
    stream = 'clear'
    @staticmethod
    def fig_names_to_fig_objects(fig_names: list[str]) -> list[Figure]: return [plt.figure(num=afig) for afig in fig_names]
    @staticmethod
    def get_all_fig_objects() -> list[Figure]: return [plt.figure(k) for k in plt.get_figlabels()]  # plt.get_fignums()
    def __init__(self, save_dir: OPLike = None, save_name: Optional[str] = None, watch_figs: Optional[list[Figure]] = None, max_calls: int = 2000, delay: int = 100, **kwargs: Any):
        """How to control what to be saved: you can either pass the figures to be tracked at init time, pass them dynamically at add time, or, add method will capture every figure and axis"""
        if watch_figs is None: tmp = []
        else: tmp = watch_figs
        self.watch_figs = tmp
        self.save_name = timestamp(name=save_name)
        self.save_dir = P(save_dir) if save_dir is not None else P.tmpdir(prefix="tmp_fig_save")
        self.kwargs, self.counter, self.delay, self.max = kwargs, 0, delay, max_calls
    def add(self, fignames: Optional[list[str]] = None, names: Optional[list[str]] = None, **kwargs: Any):  # generic method used at runtime, never changed.
        print(f"Saver added frame number {self.counter}", end='\r')
        self.counter += 1; plt.pause(self.delay * 0.001)
        _ = print('Turning off IO') if self.counter > self.max else None; plt.ioff()
        if fignames is not None:
            fig_objs = self.fig_names_to_fig_objects(fignames)
            self.watch_figs += fig_objs  # path sent explicitly, # None exist ==> add all else # they exist already.
        if names is None: names = [timestamp(name=str(a_figure.get_label())) for a_figure in self.watch_figs]  # individual save path, useful for PNG.
        for afig, aname in zip(self.watch_figs, names): self._save(afig, aname, **kwargs)
    def _save(self, *args: Any, **kwargs: Any): pass  # called by generic method `add`, to be implemented when subclassing.

class Null(GenericSave):  # serves as a filler.
    def __init__(self, *args: Any, **kwargs: Any): super().__init__(*args, **kwargs); self.fname = self.save_dir
    def finish(self): print(f"Nothing saved by {self}"); return self.fname
class PDF(GenericSave):  # For pdf, you just need any stream [update, clear, accumalate], preferabbly the fastest."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        from matplotlib.backends.backend_pdf import PdfPages
        self.fname = self.save_dir.joinpath(self.save_name + ('.pdf' if '.pdf' not in str(self.save_name) else ''))
        self.pp = PdfPages(str(self.fname))
    def _save(self, a_fig: Figure, a_name: str, bbox_inches: str = 'tight', pad_inches: float = 0.3, **kwargs: Any):
        _ = a_name
        self.pp.savefig(a_fig, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)
    def finish(self): print("Saving results ..."); self.pp.close(); print("SAVED PDF @", P(self.fname).absolute().as_uri()); return self
class PNG(GenericSave):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.fname = self.save_dir = self.save_dir.joinpath(self.save_name)
    def _save(self, afigure: Figure, aname: str, dpi: int = 150, **kwargs: Any):
        fname = self.save_dir.joinpath(validate_name(aname)).create(parents_only=True)
        afigure.savefig(str(fname), bbox_inches='tight', pad_inches=0.3, dpi=dpi, **kwargs)
    def finish(self): print("SAVED PNGs @", P(self.fname).absolute().as_uri()); return self

# class GIF(GenericSave):  # NOT RECOMMENDED, used GIFFileBased instead.
#     """This class uses ArtistAnimation: works on lines and images list attached to figure axes and Doesn't work on axes, unless you add large number of them. As such, titles are not incorporated etc (limitation).
#     Requirements: same axis must persist (If you clear the axis, nothing will be saved), only new objects are drawn inside it. Additionally, the objects drawn must not be removed, or updated, instead they should pile up in axis.
#     # usually it is smoother when adding animate=True to plot or imshow commands for GIF purpose
#     Works for images only. Add more .imshow to the same axis, and that's it. imshow will conver up previous images. For lines, it will superimpose it and will look ugly.
#     The class will automatically detect new lines by their "neo" labels and add them then hide them for the next round.
#     """
#     def __init__(self, interval: int = 100, **kwargs: Any):
#         super().__init__(**kwargs); from collections import defaultdict
#         self.container: dict[str, Any] = defaultdict(lambda: [])
#         self.interval = interval
#         self.fname = None  # determined at finish time.
#     def _save(self, afigure: Figure, aname: str, cla: bool = False, **kwargs: Any):
#         _ = aname, kwargs, cla
#         fig_list, subcontainer = self.container[afigure.get_label()], []
#         for item in FigureManager.findobj(afigure.get_label(), 'neo'): item.set_label('processed'); item.set_visible(False); subcontainer += [item]
#         fig_list.append(subcontainer)  # if you want the method coupled with cla being used in main, then it add_line is required for axes.
#     def finish(self):
#         print("Saving the GIF ....")
#         for _idx, a_fig in enumerate(self.watch_figs):
#             if ims := self.container[a_fig.get_label()]:
#                 self.fname = self.save_dir.joinpath(f'{a_fig.get_label()}_{self.save_name}.gif')
#                 ani = animation.ArtistAnimation(a_fig, ims, interval=self.interval, blit=True, repeat_delay=1000)
#                 ani.save(str(self.fname), writer=animation.PillowWriter(fps=4))  # if you don't specify the writer, it goes to ffmpeg by default then try others if that is not available, resulting in behaviours that is not consistent across machines.
#                 print(f"SAVED GIF @", P(self.fname).absolute().as_uri())
#             else: print(f"Nothing to be saved by GIF writer."); return self.fname

class GIFFileBased(GenericSave):
    def __init__(self, fps: int = 4, dpi: int = 100, bitrate: int = 1800, _type: SAVE_TYPE = 'GIFFileBased', **kwargs: Any):
        super().__init__(**kwargs); assert_requirements()
        if _type == 'GIFPipeBased':
            extension = '.gif'
            writer: Any = animation.ImageMagickFileWriter  # internally calls: matplotlib._get_executable_info("magick")
        elif _type == "GIFFileBased":
            extension = '.gif'
            writer = animation.ImageMagickWriter
        elif _type == 'MPEGFileBased':
            extension = '.mp4'
            writer = animation.FFMpegFileWriter
        elif _type == 'MPEGPipeBased':
            extension = '.mp4'
            writer = animation.FFMpegWriter
        else: raise ValueError("Unknown writer.")
        import getpass
        self.writer = writer(fps=fps, metadata={'artist': getpass.getuser()}, bitrate=bitrate)
        self.fname = self.save_dir.joinpath(self.save_name + extension)
        assert self.watch_figs, "No figure was sent during instantiation of saver, therefore the writer cannot be setup. Did you mean to use an autosaver?"
        self.writer.setup(fig=self.watch_figs[0], outfile=str(self.fname), dpi=dpi)
    def _save(self, afig: str, aname: str, **kwargs: Any) -> None:
        _ = aname, afig
        self.writer.grab_frame(**kwargs)
    def finish(self):
        print('Saving results ...')
        self.writer.finish()
        print("SAVED GIF @", P(self.fname).absolute().as_uri())
        return self
class GIFPipeBased(GIFFileBased):
    def __init__(self): super().__init__(_type='GIFPipeBased')
class MPEGFileBased(GIFFileBased):
    def __init__(self): super().__init__(_type='MPEGFileBased')
class MPEGPipeBased(GIFFileBased):
    def __init__(self): super().__init__(_type='MPEGPipeBased')


Saver: TypeAlias = Union[Null, PDF, PNG, GIFFileBased, GIFPipeBased, MPEGFileBased, MPEGPipeBased]


# =================================== AUTO SAVERS ==========================================
class GenericAuto(GenericSave):
    """Parses the data internally, hence requires artist with animate method implemetend. Artist needs to have .fig attribute."""
    save_type = 'auto'
    def __init__(self, saver: Saver, animator: Callable[[Any], Any], data: list['npt.NDArray[np.float64]'], names_list: Optional[list[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.saver = saver
        self.plotter = None
        assert_requirements()
        self.animator = animator
        self.data = data
        self.names_list = names_list
    def animate(self):
        plt.pause(0.5)  # give time for figures to show up before updating them
        from tqdm import tqdm
        for idx, datum in tqdm(enumerate(self.data)):
            self.animator(datum)
            self.saver.add(names=[self.names_list[idx] if self.names_list is not None else str(idx)])
        self.saver.finish()


# class GIFAuto(GenericSave):
#     def __init__(self, animator: Callable[[Any], Any], data: list['npt.NDArray[np.float64]'], interval: int = 500, extension: str = 'gif', fps: int = 4, metadata: Optional[dict[str, Any]] = None, **kwargs: Any):
#         super().__init__()
#         self.data = data
#         writer = animation.PillowWriter(fps=fps) if extension == '.mp4' else animation.FFMpegWriter(fps=fps, metadata=metadata or {}, bitrate=2500)
#         self.animator = animator
#         plt.pause(self.delay * 0.001)  # give time for figures to show up before updating them
#         self.ani = animation.FuncAnimation(fig=self.plotter.fig, func=self.plotter.animate, frames=(i for i in zip(*self.data)), interval=interval, repeat_delay=1500, fargs=None, cache_frame_data=True, save_count=10000)
#         self.fname = self.save_dir.joinpath(self.save_name + f".{extension}")
#         self.ani.save(filename=str(self.fname), writer=writer)
#         print(f"SAVED GIF @ ", P(self.fname).absolute().as_uri())
#     def animate(self) -> None:
#         plt.pause(interval=0.5)  # give time for figures to show up before updating them
#         for idx, datum in tqdm(enumerate(self.data)):
#             self.animator(datum)
#             self.saver.add(names=[self.names_list[idx] if self.names_list is not None else str(idx)])
#         self.saver.finish()


class PDFAuto(GenericAuto):
    """Parses the data internally, hence requires artist with animate method implemetend. Artist needs to have .fig attribute."""
    save_type = 'auto'
    def __init__(self, animator: Callable[[Any], Any], data: list['npt.NDArray[np.float64]'], names_list: Optional[list[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.data = data
        self.names_list = names_list
        self.animator = animator
        self.saver = PDF(**kwargs)
        assert_requirements()
        self.animate()
    def animate(self):
        plt.pause(0.5)  # give time for figures to show up before updating them
        from tqdm import tqdm
        for idx, datum in tqdm(enumerate(self.data)):
            self.animator(datum)
            self.saver.add(names=[self.names_list[idx] if self.names_list is not None else str(idx)])
        self.saver.finish()


# class PNGAuto(GenericSave):
#     def __init__(self, **kwargs: Any):
#         super().__init__(**kwargs)
#         self.saver = PNG(**kwargs)
#         self.save_dir = self.saver.save_dir
#         self.animate()
#         self.fname = self.saver.fname
#     def animate(self):
#         plt.pause(0.5)  # give time for figures to show up before updating them
#         from tqdm import tqdm
#         for idx, datum in tqdm(enumerate(self.data)):
#             self.animator(datum)
#             self.saver.add(names=[self.names_list[idx] if self.names_list is not None else str(idx)])
#         self.saver.finish()

class NullAuto(GenericAuto):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.saver = Null(**kwargs)
        self.fname = self.saver.fname
        self.animate()
# class GIFFileBasedAuto(GenericAuto):
#     def __init__(self, plotter_class, data: 'npt.NDArray[np.float64]', fps: int = 4, dpi: int = 150, bitrate: int = 2500, _type: str = 'GIFFileBasedAuto', **kwargs):
#         super().__init__(**kwargs)
#         if _type == 'GIFPipeBasedAuto': writer = animation.ImageMagickFileWriter; extension = '.gif'
#         elif _type == 'MPEGFileBasedAuto': writer = animation.FFMpegFileWriter; extension = '.mp4'
#         elif _type == 'MPEGPipeBasedAuto': writer = animation.FFMpegWriter; extension = '.mp4'
#         else: raise ValueError("Unknown writer.")
#         self.saver = writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
#         self.fname = self.save_dir.joinpath(self.save_name + extension)
#         self.data = lambda: (i for i in zip(*data)); self.plotter = plotter_class(*[piece[0] for piece in data], **kwargs); plt.pause(0.5); from tqdm import tqdm
#         with self.saver.saving(fig=self.plotter.fig, outfile=self.fname, dpi=dpi):
#             for datum in tqdm(self.data()): self.plotter.animate(datum); self.saver.grab_frame(); plt.pause(self.delay * 0.001)
#         print(f"SAVED GIF successfully @ {self.fname}")
# class GIFPipeBasedAuto(GIFFileBasedAuto):
#     def __init__(self, *args: Any, **kwargs: Any): super().__init__(*args, _type='GIFPipeBasedAuto', **kwargs)
# class MPEGFileBasedAuto(GIFFileBasedAuto):
#     def __init__(self, *args: Any, **kwargs: Any): super().__init__(*args, _type='MPEGFileBasedAuto', **kwargs)
# class MPEGPipeBasedAuto(GIFFileBasedAuto):
#     def __init__(self, *args: Any, **kwargs: Any): super().__init__(*args, _type='MPEGPipeBasedAuto', **kwargs)

# from matplotlib.colors import Colormap
# q = Colormap()
# mcolors.


class FigureManager:
    """Base class for Artist & Viewers to give it free animation powers"""
    def __init__(self, info_loc: Optional[tuple[float, float]] = None, figpolicy: FigurePolicy = FigurePolicy.same):
        self.figpolicy = figpolicy
        self.fig: Optional[Figure] = None
        self.ax: Optional[list[Axes]] = None
        self.event: Optional[Event] = None
        self.cmaps: Cycle[str] = Cycle(PLT_CMAPS)
        self.colors: Cycle[str] = Cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        self.mcolors: list[str] = list(CSS4_COLORS.keys())
        tmp: list[str] = [str(item) for item in CSS4_COLORS.values()]
        self.facecolor: Cycle[str] = Cycle(iterable=tmp)
        self.cmaps.set_value('viridis')
        self.idx_cycle = Cycle([])
        self.pause: bool = False  # animation
        self.help_menu = {
            '_-=+[{]}\\': {
            'help': "Adjust Vmin Vmax. Shift + key applies change to all axes \\ toggles auto-brightness ", 'func': self.adjust_brightness},
            "/": {'help': 'Show/Hide info text', 'func': self.text_info},
            "h": {'help': 'Show/Hide help menu', 'func': self.show_help},
            "tyTY": {'help': 'Change color map', 'func': self.change_cmap},
            '<>': {'help': 'Change figure face color', 'func': self.change_facecolor},
            "v": {'help': 'Show/Hide pixel values (Toggle)', 'func': self.show_pix_val},
            'P': {'help': 'Pause/Proceed (Toggle)', 'func': self.pause_func},
            'r': {'help': 'Replay', 'func': self.replay},
            '1': {'help': 'Previous Image', 'func': self.previous}, '2': {'help': 'Next Image', 'func': self.next},
            'S': {'help': 'Save Object', 'func': self.save},
            'c': {'help': 'Show/Hide cursor', 'func': self.show_cursor},
            'aA': {'help': 'Show/Hide ticks and their labels', 'func': self.show_ticks},
            'alt+a': {'help': 'Show/Hide annotations', 'func': FigureManager.toggle_annotate}}  # IMPORTANT: add the 'alt/ctrl+key' versions of key after the key in the dictionary above, not before, otherwise the naked key version will statisfy the condition `is key in this`? in the parser.
        self.auto_brightness, self.pix_vals = False, False; self.boundaries_flag, self.annot_flag = True, False
        self.info_loc = (0.8, 0.01) if info_loc is None else info_loc
        self.message = ''
        self.message_obj: Optional[Any] = None
        self.cursor = None
    def show_help(self, event: Any):
        default_plt = {"q ": {'help': "Quit Figure."},
                       "Ll": {'help': "change x/y scale to log and back to linear (toggle)"},
                       "Gg": {'help': "Turn on and off x and y grid respectively."},
                       "s ": {'help': "Save Figure"},
                       "f ": {'help': "Toggle Full screen"},
                       "p ": {'help': "Select / Deselect Pan"}}
        if "Keyboard shortcuts" in plt.get_figlabels(): plt.close("Keyboard shortcuts"); _ = event  # toggle
        else:
            fig = plt.figure(num="Keyboard shortcuts")
            for i, key in enumerate(self.help_menu.keys()): fig.text(0.1, 1 - 0.05 * (i + 1), f"{key:30s} {self.help_menu[key]['help']}")
            print(pd.DataFrame([[val['help'], key] for key, val in self.help_menu.items()], columns=['Action', 'Key']), "\nDefault plt Keys:\n")
            print(pd.DataFrame([[val['help'], key] for key, val in default_plt.items()], columns=['Action', 'Key']))
    # =============== EVENT METHODS ====================================
    def animate(self):
        raise NotImplementedError("a method of the artist child class that is inheriting from this class to define behaviour when user press next or previous buttons.")
    def connect(self):
        if self.fig is not None: self.fig.canvas.mpl_connect('key_press_event', self.process_key); return self
    def process_key(self, event: Any):
        self.event = event  # useful for debugging.
        # event.
        for key, value in self.help_menu.items():
            if event.key in key:
                value['func'](event)  # type: ignore
                fig = self.fig
                if fig is not None:
                    if self.message_obj: self.message_obj.remove()
                    self.message_obj = fig.text(*self.info_loc, self.message, fontsize=8)
                else: print("Cant update info text, figure is not defined yet.")
                break
        if event.key != 'q': event.canvas.figure.canvas.draw()  # for smooth quit without throwing errors  # don't update if you want to quit.
    def toggle_annotate(self, event: Any):
        self.annot_flag = not self.annot_flag
        if event.inaxes and event.inaxes.imageevent.inaxes.images[0].set_picker(True):
            self.message = f"Annotation flag is toggled to {self.annot_flag}"
    # def annotate(self, event: Any, axis: Optional[Axes] = None, data: Optional[list[Any]] = None):
    #     self.event = event; e = event.mouseevent; ax = e.inaxes if axis is None else axis
    #     if not ax: return None
    #     if not hasattr(ax, 'annot_obj'):
    #         ax.annot_obj = ax.annotate("", xy=(0, 0), xytext=(-30, 30), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="w", connectionstyle="arc3"),
    #                                    va="bottom", ha="left", fontsize=10, bbox=dict(boxstyle="round", fc="w"), )
    #     else: ax.annot_obj.set_visible(self.annot_flag)
    #     x, y = int(np.round(e.xdata)), int(np.round(e.ydata))
    #     z = e.inaxes.images[0].get_array()[y, x] if data is None else data[y, x]
    #     ax.annot_obj.set_text(f'x:{x}\ny:{y}\nvalue:{z:.3f}')
    #     ax.annot_obj.xy = (x, y)
    #     self.fig.canvas.draw_idle()
    def save(self, event: Any):
        _ = event
        Save.pickle(path=P.tmpfile(name="figure_manager"), obj=self)
    def replay(self, event: Any):
        _ = event
        self.pause = False
        self.idx_cycle.set_index(0)
        self.message = 'Replaying'; self.animate()
    def pause_func(self, event: Any):
        _ = event
        self.pause = not self.pause
        self.message = f'Pause flag is set to {self.pause}'
        self.animate()
    def previous(self, event: Any):
        _ = event
        self.idx_cycle.previous()
        self.message = f'Previous {self.idx_cycle}'
        self.animate()
    def next(self, event: Any):
        _ = event
        self.idx_cycle.next()
        self.message = f'Next {self.idx_cycle}'
        self.animate()
    def text_info(self, event: Any):
        _ = event
        self.message = ''
    def change_facecolor(self, event: Any):
        assert self.fig is not None, "Figure is not defined yet."
        if event.key == '>': tmp = self.facecolor.next()
        else: tmp = self.facecolor.previous()
        self.fig.set_facecolor(tmp)
        self.message = f"Figure facecolor was set to {self.mcolors[self.facecolor.get_index()]}"
    def adjust_brightness(self, event: Any):
        ax, message = event.inaxes, "None"
        if ax is None or not ax.images: return None
        if event.key == '\\':
            self.auto_brightness = not self.auto_brightness; message = f"Auto-brightness flag is toggled to {self.auto_brightness}"
            if self.auto_brightness:
                if self.ax is not None:
                    im = self.ax[0].images[0]
                    im.norm.autoscale(im.get_array())  # type: ignore  # changes to all ims take place in animate as in ImShow and Nifti methods animate.
        vmin, vmax = ax.images[0].get_clim()
        if event.key in '-_': message = 'increase vmin'; vmin += 1
        elif event.key in '[{': message = 'decrease vmin'; vmin -= 1
        elif event.key in '=+': message = 'increase vmax'; vmax += 1
        elif event.key in ']}': message = 'decrease vmax'; vmax -= 1
        self.message = message + '  ' + str(round(vmin, 1)) + '  ' + str(round(vmax, 1))
        if event.key in '_+}{' and self.fig is not None:
            for ax in self.fig.axes:
                if ax.images: ax.images[0].set_clim((vmin, vmax))  # type: ignore
        else:
            if ax.images: ax.images[0].set_clim((vmin, vmax))
    def change_cmap(self, event: Any):
        ax = event.inaxes
        if ax is not None:
            assert self.fig is not None, "Figure is not defined yet."
            cmap = self.cmaps.next() if event.key in 'tT' else self.cmaps.previous()
            if event.key in 'TY':
                for ax in self.fig.axes:
                    for im in ax.images: im.set_cmap(cmap)  # type: ignore
            else:
                for im in ax.images: im.set_cmap(cmap)
            self.message = f"Color map changed to {ax.images[0].cmap.name}"  # type: ignore
    def show_pix_val(self, event: Any):
        ax = event.inaxes
        if ax is not None:
            self.pix_vals = not self.pix_vals
            self.message = f"Pixel values flag set to {self.pix_vals}"
            if self.pix_vals: self.show_pixels_values(ax)
            else:
                while len(ax.texts) > 0: _ = [text.remove() for text in ax.texts]
    def show_cursor(self, event: Any):
        ax = event.inaxes
        if not ax: return None  # don't do this if c was pressed outside an axis.
        if hasattr(ax, 'cursor_'):  # is this the first time?
            if ax.cursor_ is None: ax.cursor_ = widgets.Cursor(ax=ax, vertOn=True, horizOn=True, color='red', lw=1.0)
            else: ax.cursor_ = None  # toggle the cursor.
            self.message = f'Cursor flag set to {bool(ax.cursor_)}'
        else: ax.cursor_ = None; self.show_cursor(event)  # first call
    def show_ticks(self, event: Any):
        self.boundaries_flag = not self.boundaries_flag
        axis = event.inaxes
        if event.key == 'a' and axis:
            FigureManager.toggle_ticks(axis)
            self.message = f"Boundaries flag set to {self.boundaries_flag} in {axis}"
        elif self.ax is not None:
            for ax in self.ax: FigureManager.toggle_ticks(ax)
    # ====================== class methods ===============================
    @staticmethod
    def get_fig(figpolicy: FigurePolicy, figname: str = '', suffix: Optional[str] = None, **kwargs: Any):
        exist = True if figname in plt.get_figlabels() else False
        if figpolicy is FigurePolicy.same: return plt.figure(num=figname, **kwargs)
        elif figpolicy is FigurePolicy.add_new: return plt.figure(num=(timestamp(name=figname) if suffix is None else figname + suffix) if exist else figname, **kwargs)
        elif figpolicy is FigurePolicy.close_create_new:
            if exist: plt.close(figname)
            return plt.figure(num=figname, **kwargs)
    @staticmethod
    def maximize_fig() -> None:
        fig_manager = plt.get_current_fig_manager()
        if fig_manager is not None:
            if hasattr( fig_manager, "full_screen_toggle"): fig_manager.full_screen_toggle()
            elif hasattr(fig_manager, "window"):
                fig_manager.resize(*fig_manager.window.maxsize())  # type: ignore

    @staticmethod
    def clear_axes(axes: Optional[list[Axes]]) -> None:
        if axes is not None:
            for ax in axes:
                ax.cla()
    @staticmethod
    def transperent_fig(fig: Optional[Figure]) -> None:
        assert fig is not None, "Figure is not defined yet."
        fig.canvas.manager.window.attributes("-transparentcolor", "white")  # type: ignore
    # @staticmethod
    # def close(fig: Optional[Figure]): plt.close(fig)
    # ====================== axis helpers ========================
    @staticmethod
    def grid(ax: Axes, factor: int = 5, x_or_y: Literal['x', 'y', 'both'] = 'both', color: str = 'gray', alpha1: float = 0.5, alpha2: float = 0.25):
        ax.grid(which='major', axis='x', color='gray', linewidth=0.5, alpha=alpha1); ax.grid(which='major', axis='y', color='gray', linewidth=0.5, alpha=alpha1)
        if x_or_y in {'both', 'x'}: xt = ax.get_xticks(); ax.xaxis.set_minor_locator(MultipleLocator((xt[1] - xt[0]) / factor)); ax.grid(which='minor', axis='x', color=color, linewidth=0.5, alpha=alpha2)
        if x_or_y in {'both', 'y'}: yt = ax.get_yticks(); ax.yaxis.set_minor_locator(MultipleLocator((yt[1] - yt[0]) / factor)); ax.grid(which='minor', axis='y', color=color, linewidth=0.5, alpha=alpha2)
    @staticmethod
    def set_ax_size(ax: Axes, w: float, h: float, units: Literal["inches", "pixels"] = 'inches'):
        left, r, t, b = ax.figure.subplotpars.left, ax.figure.subplotpars.right, ax.figure.subplotpars.top, ax.figure.subplotpars.bottom  # type: ignore
        _ = units
        if isinstance(ax.figure, Figure): ax.figure.set_size_inches(float(w) / (r - left), float(h) / (t - b))
    @staticmethod
    def get_ax_size(ax: Axes, units: Literal["inches", "pixels"] = "inches"):
        assert ax.figure is not None, "Figure is not defined yet."
        w, h = ax.figure.get_size_inches()  # type: ignore
        if units == "pixels":
            w, h = ax.figure.get_size_inches() * ax.figure.dpi  # type: ignore
        width, height = ax.figure.subplotpars.right - ax.figure.subplotpars.left, ax.figure.subplotpars.top - ax.figure.subplotpars.bottom  # type: ignore
        return w * width, h * height
    @staticmethod
    def toggle_ticks(an_ax: Axes, state: Optional[bool] = None):
        lines = an_ax.get_yticklines() + an_ax.get_xticklines() + an_ax.get_xticklabels() + an_ax.get_yticklabels()
        for line in lines:
            flag = not line.get_visible() if state is None else state
            line.set_visible(flag)
    @staticmethod
    def set_ax_to_real_life_size(ax: Axes, inch_per_unit: float = 1 / 25.4):
        FigureManager.set_ax_size(ax, (ax.get_xlim()[1] - ax.get_xlim()[0]) * inch_per_unit, (ax.get_ylim()[1] - ax.get_ylim()[0]) * inch_per_unit)
    @staticmethod
    def show_pixels_values(ax: Axes):
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        if ymin > ymax: ymin, ymax = ymax, ymin  # default imshow settings
        for (j, i), label in np.ndenumerate(ax.images[0].get_array()):  # type: ignore
            if (xmin < i < xmax) and (ymin < j < ymax):
                ax.text(i, j, np.round(label).__int__(), ha='center', va='center', size=8)
    # ============================ matplotlib setup ==============================
    @staticmethod
    def get_nrows_ncols(num_plots: int, nrows: Optional[int] = None, ncols: Optional[int] = None) -> tuple[int, int]:
        if not nrows and not ncols:
            nrows_res, ncols_res = int(np.floor(np.sqrt(num_plots))), int(np.ceil(np.sqrt(num_plots)))
            while nrows_res * ncols_res < num_plots: ncols_res += 1
        elif not ncols and nrows:
            ncols_res = int(np.ceil(num_plots / nrows))
            nrows_res = nrows
        elif not nrows and ncols:
            nrows_res = int(np.ceil(num_plots / ncols))
            ncols_res = ncols
        else: raise ValueError("Both nrows and ncols are defined, which is not allowed.")
        return nrows_res, ncols_res
    @staticmethod
    def findobj(figname: str, obj_name: str):
        return plt.figure(num=figname).findobj(lambda x: x.get_label() == obj_name)
    @staticmethod
    def try_figure_size() -> None:
        fig, ax = plt.subplots()
        x = np.arange(0, 100, 0.01)
        y = np.sin(x) * 100
        ax.plot(x, y); ax.axis("square"); ax.set_xlim(0, 100); ax.set_ylim(-100, 100)
        FigureManager.set_ax_to_real_life_size(ax); fig.savefig(str(P.tmp() / "trial.png"), dpi=250)
    @staticmethod
    def write(txt: str, name: str = "text", size: int = 8, **kwargs: Any) -> Figure:
        fig = plt.figure(figsize=(11.69, 8.27), num=name)
        FigureManager.maximize_fig()
        fig.clf()
        # transform=fig.transFigure
        fig.text(0.5, 0.5, txt, size=size, ha="center", va='center', **kwargs)
        return fig
    @staticmethod
    def activate_latex(size: int = 20) -> None:
        plt.rc('xtick', labelsize=size); plt.rc('ytick', labelsize=size)
        plt.rc('axes', titlesize=size); plt.rc('legend', fontsize=size / 1.5)  # rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True; plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    @staticmethod
    def set_linestyles_and_markers_and_colors(test: bool = False):
        from cycler import cycler; from matplotlib import lines
        markers = list(lines.lineMarkers.keys())[:-4]  # ignore the None
        linestyles = (list(lines.lineStyles.keys())[:-3] * 10)[:len(markers)]
        colors = (plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10)[:len(markers)]
        default_cycler = (cycler(linestyle=linestyles) + cycler(marker=markers) + cycler(color=colors))
        plt.rc('axes', prop_cycle=default_cycler)
        if test: _ = [plt.plot(aq + idx * 2) for idx, aq in enumerate(np.random.randn(10, 10))]


class VisibilityViewer(FigureManager):
    """
    This is used for browsing purpose, as opposed to saving.
    This class works on hiding axes shown on a plot, so that a new plot can be drawn.
    Once the entire loop is finished, you can browse through the plots with the keyboard Animation linked to `animate`
    Downside: slow if number of axes, lines, texts, etc. are too large. In that case, it is better to avoid this viewer and plot on the fly during animation.
    """
    artist: ARTIST = 'external'
    parser: PARSER = 'external'
    stream: STREAM = 'accumulate'
    def __init__(self, fig: Figure):
        super().__init__()
        self.objs_repo: list[list[Any]] = []  # list of lists of axes and texts.
        self.fig = fig
        self.connect()
        self.idx_cycle = Cycle([])
    def add(self, objs: list[Any]) -> None:
        self.idx_cycle.expand(val=len(self.idx_cycle.list))
        self.objs_repo.append(objs)
        for obj in (self.objs_repo[-2] if len(self.objs_repo) > 1 else []): obj.set_visible(False)
        print(f"{self.__class__.__name__} added plot number {self.idx_cycle.get_index()}", end='\r')
    def animate(self):
        for an_obj in self.objs_repo[self.idx_cycle.prev_index]: an_obj.set_visible(False)
        visible_objs = self.objs_repo[self.idx_cycle.get_index()]
        for an_obj in visible_objs:
            an_obj.set_visible(True)
        x, y = None, None
        for an_obj in visible_objs:
            if an_obj.__class__.__name__ == 'Line2D':
                x, y = an_obj.get_xdata(), an_obj.get_ydata()
                break
        fig = self.fig
        if fig is not None:
            if x is not None:
                fig.axes[0].set_xlim(min(x), max(x))
            if y is not None:
                fig.axes[0].set_ylim(min(y), max(y))
            fig.canvas.draw()
    @staticmethod
    def usage_example():
        with plt.style.context(style='seaborn-v0_8-dark'):
            fig = plt.figure()
            FigureManager.maximize_fig()
            ax = fig.subplots()
            assert isinstance(ax, Axes)
            viewer = VisibilityViewer(fig)
            for _i in range(10):
                data1 = np.random.randn(100)
                obj = ax.plot(data1, label=f'plot number {_i}')
                viewer.add(obj)
            FigureManager.grid(ax)
            viewer.animate()
            plt.show()


class LineArtist(FigureManager):
    artist: ARTIST = 'internal'  # This object knows how to draw a figure from curve-type data.
    parser: PARSER = 'external'
    stream: STREAM = 'accumulate'
    def __init__(self, ax: Optional[Axes] = None, figname: str = 'Graph', title: str = '', label: str = 'curve', style: str = 'seaborn-v0_8-dark', figpolicy: FigurePolicy = FigurePolicy.add_new, figsize: tuple[int, int] = (14, 8)):
        super().__init__(figpolicy=figpolicy)
        self.style = style
        self.title = title
        self.line: Optional[list[Any]] = None
        self.cursor = None
        self.check_b = None
        if ax is None:  # create a figure
            with plt.style.context(style=self.style):  # type: ignore
                self.fig = FigureManager.get_fig(figpolicy=self.figpolicy, figname=figname, figsize=figsize)
                tmp = self.fig.subplots()
                assert tmp is not None, "Subplots failed to create axes."
                if isinstance(tmp, list): self.ax = tmp
                elif isinstance(tmp, Axes):
                    self.ax = [tmp]
        else:
            self.ax = [ax]
            self.fig: Figure = ax.figure or None  # type: ignore
        self.visibility_ax: list[float] = [0.01, 0.05, 0.2, 0.15]
        self.txt: list[str] = []
        self.label: str = label
    def animate(self):
        pass
    def plot(self, *args: Any, legends: Optional[list[str]] = None, title: Optional[str] = None, **kwargs: Any) -> None:
        assert self.ax is not None, "Axes is not defined yet."
        for ax in self.ax:
            self.line = ax.plot(*args, **kwargs)
            ax.legend(legends or [])
            if title is not None: ax.set_title(title)
            ax.grid(visible=True)
    def plot_dict(self, adict: dict[str, Any], title: str = '', xlabel: str = '', ylabel: str = ''):
        for key, val in adict.items(): self.plot(val, label=key)
        assert self.ax is not None, "Axes is not defined yet."
        for ax in self.ax:
            ax.legend()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        return self
    def plot_twin(self, c1: Any, c2: Any, x: Optional[Any] = None, l1: str = '', l2: str = '', ax: Optional[Axes] = None) -> None:
        if ax is None:
            if self.ax is not None:
                ax = self.ax[0]
            else: raise ValueError("No axes to plot on.")
        else: raise ValueError("Not implemented yet.")
        twin_ax = ax.twinx()
        line1 = ax.plot(x or range(len(c1)), c1, color="blue", label=l1)[0]
        line2 = twin_ax.plot(c2, color="red", label=l2)[0]  # type: ignore
        twin_ax.legend([line1, line2], [l1, l2])  # type: ignore
        ax.set_ylabel(l1); twin_ax.set_ylabel(l2)
        plt.show()
    def suptitle(self, title: str) -> None:
        assert self.fig is not None, "Figure is not defined yet."
        self.txt = [str(self.fig.text(0.5, 0.98, title, ha='center', va='center', size=9))]
    def clear(self):
        assert self.fig is not None, "Figure is not defined yet."
        self.fig.clf()  # objects that use this class will demand that the artist expose this method, in addition to .plot()
    def axes(self): return self.ax  # objects that use this class will demand that the artist expose this method, in addition to .plot()
    # def visibility(self):
    #     from matplotlib.widgets import CheckButtons
    #     self.fig.subplots_adjust(left=0.3); self.visibility_ax[-1] = 0.05 * len(self.ax.lines)
    #     rax = self.fig.add_axes(self.visibility_ax)
    #     labels, visibility = [str(line.get_label()) for line in self.ax.lines], [line.get_visible() for line in self.ax.lines]
    #     self.check_b = CheckButtons(rax, labels, visibility)
    #     def func(label): index = labels.index(label); self.ax.lines[index].set_visible(not self.ax.lines[index].get_visible()); self.fig.canvas.draw()
    #     self.check_b.on_clicked(func)
    # @staticmethod
    # def styler(plot_gen):
    #     for astyle in plt.style.available:
    #         with plt.style.context(style=astyle): plot_gen(); plt.title(astyle); plt.pause(1); plt.cla()


# class VisibilityViewerAuto(VisibilityViewer):
#     artist: ARTIST = 'external'
#     parser: PARSER = 'internal'
#     stream: STREAM = 'update'
#     def __init__(self, data: Optional['npt.NDArray[np.float64]'] = None, artist: Optional[Artist] = None, stream: STREAM = 'clear', save_type: Type[Saver] = Null, save_dir: OPLike = None, save_name: Optional[str] = None, delay: int = 1,
#                  titles: Optional[list[str]] = None, legends: Optional[list[str]] = None, x_labels: Optional[list[str]] = None, pause: bool = True, **kwargs: Any):
#         """data: tensor of form  NumInputsForAnimation x ArgsPerPlot (to be unstarred) x Input (possible points x signals)
#         stream: ensure that behaviour of artist is consistent with stream. When `cccumulate`, artist should create new axes whenever plot is called."""
#         self.data = data
#         self.artist: Artist = artist or Artist()
#         self.legends = [f"Plot {i}" for i in range(len(self.data))] if legends is None else legends; self.titles = titles if titles is not None else np.arange(len(self.data))
#         super().__init__(fig=self.artist.fig); _ = kwargs
#         self.saver = save_type(watch_figs=[self.artist.fig], save_dir=save_dir, save_name=save_name, delay=delay, fps=1000 / delay)
#         self.index_max, self.pause, self.stream, self.lables = len(self.data), pause, stream, x_labels
#     test = staticmethod(lambda: VisibilityViewerAuto(data=np.random.randn(10, 1, 10, 3)))
#     def animate(self):
#         for i in range(0, self.index_max):
#             self.artist.plot(*self.data[i], title=self.titles[i], legends=self.legends)  # replot the new data point on a new axis.
#             super().add(self.artist.axes()) if self.stream == 'accumulate' else self.artist.clear()
#             self.saver.add()
#             if self.pause: break
#             else: self.idx_cycle.set_index(i)
#         return self.saver.finish()  # arrived at last image and not in manual mode


class ImShow(FigureManager):
    artist: ARTIST = 'internal'
    parser: PARSER = 'internal'
    stream: STREAM = 'update'
    def __init__(self, img_tensor: 'npt.NDArray[np.float64]', sup_titles: Optional[list[str]] = None, sub_labels: Optional[list[list[str]]] = None, save_type: Type[Saver] = Null, save_name: Optional[str] = None,
                 save_dir: OPLike = None, save_kwargs: Optional[dict[str, Any]] = None,
                 subplots_adjust: Any = None, gridspec: Any = None, tight: bool = True, info_loc: Optional[tuple[float, float]] = None, nrows: Optional[int] = None, ncols: Optional[int] = None, ax: Optional[Axes] = None,
                 figsize: Optional[tuple[int, int]] = None, figname: str = 'im_show', auto_brightness: bool = True, delay: int = 200, pause: bool = False, **kwargs: Any):
        """
        :param img_tensor: size N x M x W x H [x C]  # M used spatially, N for animation.
        :param sup_titles: Titles for frames (N)
        :param sub_labels: M x N. If shape sent is M
        """
        _ = save_dir, save_kwargs, save_type, save_name
        super().__init__(figpolicy=FigurePolicy.add_new)
        n, m = len(img_tensor), len(img_tensor[0])
        self.m, self.n = m, n
        super(ImShow, self).__init__(info_loc=info_loc)
        nrows, ncols = self.get_nrows_ncols(m, nrows, ncols)
        self.img_tensor = img_tensor
        self.sub_labels = sub_labels if sub_labels is not None else [[f"{i}-{j}" for j in range(m)] for i in range(n)]
        self.sup_titles = sup_titles if sup_titles is not None else [str(item) for item in np.arange(start=0, stop=n)]
        self.pause, self.kwargs, self.delay, self.auto_brightness = pause, kwargs, delay, auto_brightness
        self.fname = self.event = None
        self.ims: list[AxesImage] = []  # container for images.
        self.cmaps = Cycle(PLT_CMAPS); self.cmaps.set_value('viridis')
        if ax is None:
            self.fig = FigureManager.get_fig(figpolicy=self.figpolicy, figname=figname, figsize=(14, 9) if figsize is None else figsize, facecolor='white')
            if figsize is None: FigureManager.maximize_fig()
            if gridspec is not None:
                assert self.fig is not None
                gs = self.fig.add_gridspec(gridspec[0])
                tmp: list[Axes] = []
                for ags in gridspec[1:]:
                    assert self.fig is not None
                    qq = self.fig.add_subplot(gs[ags[0], ags[1]])
                    tmp.append(qq)
                self.ax = tmp
            else:
                tmp: list[Axes] = list(self.fig.subplots(nrows=nrows, ncols=ncols))  # type: ignore
                self.ax = tmp
        else:
            self.ax = [ax]
            fig = ax.figure
            if isinstance(fig, Figure): self.fig = fig
            raise ValueError("Figure is not defined yet.")
        # if nrows == 1 and ncols == 1: pass
        # else: self.ax = self.ax.ravel()  # make a list out of it or # make a 1D  list out of a 2D array.
        self.connect()
        # self.fig.canvas.mpl_connect("pick_event", self.annotate)
        if tight and self.fig is not None: self.fig.tight_layout()  # type: ignore
        if subplots_adjust is not None and self.fig is not None: self.fig.subplots_adjust(**subplots_adjust)  # type: ignore
        # self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name, delay=delay, fps=1000 / delay, **({} if save_kwargs is None else save_kwargs))
        if isinstance(self.ax, list):  # type: ignore
            for an_ax in self.ax: FigureManager.toggle_ticks(an_ax, state=False)
        self.idx_cycle = Cycle([item for item in range(len(self.img_tensor))])  # self.animate()
    def animate(self):
        for i in range(self.idx_cycle.get_index(), self.n):
            assert self.ax is not None, "Axes is not defined yet."
            for j, (an_image, a_label, an_ax) in enumerate(zip(self.img_tensor[i], self.sub_labels[i], self.ax)):  # with zipping, the shortest of the three, will stop the loop.
                if i == 0 and self.ims.__len__() < self.m: self.ims.append(an_ax.imshow(an_image, animated=True, **self.kwargs))
                else: self.ims[j].set_data(an_image)
                if self.auto_brightness: self.ims[j].norm.autoscale(an_image)
                an_ax.set_xlabel(f'{a_label}')
            assert isinstance(self.fig, Figure)
            self.fig.suptitle(self.sup_titles[i], fontsize=8)
            # self.saver.add(names=[self.sup_titles[i]])
            if self.pause: break
            else: self.idx_cycle.set_index(i)
        if self.idx_cycle.get_index() == self.n - 1 and not self.pause:
            # self.fname = self.saver.finish()  # arrived at last image and not in manual mode
            pass
    # @staticmethod
    # def try_cmaps(im: 'npt.NDArray[np.float64]', nrows: int = 3, ncols: int = 7, **kwargs: Any):
    #      _ = ImShow(*np.array_split([plt.get_cmap(style)(im) for style in plt.colormaps()], nrows * ncols), nrows=nrows, ncols=ncols,
    #                 sub_labels=np.array_split(plt.colormaps(), nrows * ncols), **kwargs); return [plt.get_cmap(style)(im) for style in plt.colormaps()]
    # def annotate(self, event: Any, axis: Optional[Axes] = None):
    #     _ = [super().annotate(event, axis=ax, data=ax.images[0].get_array()) for ax in self.ax]
    # @staticmethod
    # def from_img_paths(paths: list[PLike], **kwargs: Any): ImShow(List(paths).apply(plt.imread), sub_labels=List(paths).apply(lambda x: P(x).stem), **kwargs)
    # @staticmethod
    # def from_complex(data: 'npt.NDArray[np.float64]', pause: bool = True, **kwargs: Any): ImShow(data.real, data.imag, np.angle(data), abs(data), labels=['Real Part', 'Imaginary Part', 'Angle in Radians', 'Absolute Value'], pause=pause, **kwargs)
    @staticmethod
    def test() -> None: ImShow(img_tensor=np.random.rand(12, 10, 80, 120, 3))  # https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
    @staticmethod
    def resize(path: PLike, m: int, n: int):
        from crocodile.core import install_n_import
        res = install_n_import(library="skimage", package="scikit-image").transform.resize(plt.imread(str(path)), (m, n), anti_aliasing=True)
        plt.imsave(str(path), res)


if __name__ == '__main__':
    pass
