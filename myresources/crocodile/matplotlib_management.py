

from crocodile.core import List, timestamp, Save, install_n_import, validate_name
from crocodile.file_management import P
from crocodile.meta import Terminal

import matplotlib.pyplot as plt
from crocodile.msc.odds import Cycle
from matplotlib import widgets
import matplotlib.colors as mcolors
from matplotlib import animation
import enum
import subprocess
import platform
import pandas as pd
import numpy as np


"""TODO: add implementation https://github.com/gustavovelascoh/plot_update
"""


class FigurePolicy(enum.Enum):
    close_create_new = 'Close the previous figure that has the same figname and create a new fresh one'
    add_new = 'Create a new figure with same path but with added suffix'
    same = 'Grab the figure of the same path'


def assert_requirements():
    try: subprocess.check_output(['where.exe' if platform.system() == 'Windows' else 'which', 'magick'])
    except (FileNotFoundError, subprocess.CalledProcessError):
        # P(r"https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z").download().unzip().search()[0].rename("ffmpeg").move(r"C://")  # P("C:\\ffmpeg\\bin")  # add to PATH
        print("Installinng image magick")
        if __import__("platform").system() == "Windows":
            Terminal().run("winget install ImageMagick.ImageMagick", shell="powershell")  # gives ffmpeg as well
            print("You might need to restart your machine before PATH change impact takes place.")
        else: raise NotImplementedError


class FigureSave:
    class GenericSave:
        stream = ['clear', 'accumulate', 'update'][0]
        def __init__(self, save_dir=None, save_name=None, watch_figs: list or None = None, max_calls=2000, delay=100, **kwargs):
            """How to control what to be saved: you can either pass the figures to be tracked at init time, pass them dynamically at add time, or, add method will capture every figure and axis"""
            self.watch_figs = watch_figs if watch_figs is None else ([plt.figure(num=afig) for afig in watch_figs] if type(watch_figs[0]) is str else watch_figs)
            self.save_name, self.save_dir = timestamp(name=save_name), save_dir or P.tmpdir(prefix="tmp_fig_save")
            self.kwargs, self.counter, self.delay, self.max = kwargs, 0, delay, max_calls
        def add(self, fignames=None, names=None, **kwargs):  # generic method used at runtime, never changed.
            print(f"Saver added frame number {self.counter}", end='\r')
            self.counter += 1; plt.pause(self.delay * 0.001); print('Turning off IO') if self.counter > self.max else None; plt.ioff()
            self.watch_figs = [plt.figure(figname) for figname in fignames] if fignames else ([plt.figure(k) for k in plt.get_figlabels()] if self.watch_figs is None else self.watch_figs)  # path sent explicitly, # None exist ==> add all else # they exist already.
            if names is None: names = [timestamp(name=a_figure.get_label()) for a_figure in self.watch_figs]  # individual save path, useful for PNG.
            for afig, aname in zip(self.watch_figs, names): self._save(afig, aname, **kwargs)
        def _save(self, *args, **kwargs): pass  # called by generic method `add`, to be implemented when subclassing.
    class Null(GenericSave):  # serves as a filler.
        def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self.fname = self.save_dir
        def finish(self): print(f"Nothing saved by {self}"); return self.fname
    class PDF(GenericSave):  # For pdf, you just need any stream [update, clear, accumalate], preferabbly the fastest."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from matplotlib.backends.backend_pdf import PdfPages
            self.fname = self.save_dir.joinpath(self.save_name + ('.pdf' if '.pdf' not in str(self.save_name) else '')); self.pp = PdfPages(self.fname)
        def _save(self, a_fig, a_name, bbox_inches='tight', pad_inches=0.3, **kwargs): self.pp.savefig(a_fig, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)
        def finish(self): print(f"Saving results ..."); self.pp.close(); print(f"SAVED PDF @", P(self.fname).absolute().as_uri()); return self
    class PNG(GenericSave):
        def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self.fname = self.save_dir = self.save_dir.joinpath(self.save_name)
        def _save(self, afigure, aname, dpi=150, **kwargs):  afigure.savefig(self.save_dir.joinpath(validate_name(aname)).create(parents_only=True), bbox_inches='tight', pad_inches=0.3, dpi=dpi, **kwargs)
        def finish(self): print(f"SAVED PNGs @", P(self.fname).absolute().as_uri()); return self
    class GIF(GenericSave):  # NOT RECOMMENDED, used GIFFileBased instead.
        """This class uses ArtistAnimation: works on lines and images list attached to figure axes and Doesn't work on axes, unless you add large number of them. As such, titles are not incorporated etc (limitation).
        Requirements: same axis must persist (If you clear the axis, nothing will be saved), only new objects are drawn inside it. Additionally, the objects drawn must not be removed, or updated, instead they should pile up in axis.
        # usually it is smoother when adding animate=True to plot or imshow commands for GIF purpose
        Works for images only. Add more .imshow to the same axis, and that's it. imshow will conver up previous images. For lines, it will superimpose it and will look ugly.
        The class will automatically detect new lines by their "neo" labels and add them then hide them for the next round.
        """
        def __init__(self, interval=100, **kwargs):
            super().__init__(**kwargs); from collections import defaultdict
            self.container, self.interval, self.fname = defaultdict(lambda: []), interval, None  # determined at finish time.
        def _save(self, afigure, aname, cla=False, **kwargs):
            fig_list, subcontainer = self.container[afigure.get_label()], []
            for item in FigureManager.findobj(afigure, 'neo'): item.set_label('processed'); item.set_visible(False); subcontainer += [item]
            fig_list.append(subcontainer)  # if you want the method coupled with cla being used in main, then it add_line is required for axes.
        def finish(self):
            print("Saving the GIF ....")
            for idx, a_fig in enumerate(self.watch_figs):
                if ims := self.container[a_fig.get_label()]:
                    self.fname = self.save_dir.joinpath(f'{a_fig.get_label()}_{self.save_name}.gif')
                    ani = animation.ArtistAnimation(a_fig, ims, interval=self.interval, blit=True, repeat_delay=1000)
                    # noinspection PyTypeChecker
                    ani.save(self.fname, writer=animation.PillowWriter(fps=4))  # if you don't specify the writer, it goes to ffmpeg by default then try others if that is not available, resulting in behaviours that is not consistent across machines.
                    print(f"SAVED GIF @", P(self.fname).absolute().as_uri())
                else: print(f"Nothing to be saved by GIF writer."); return self.fname
    class GIFFileBased(GenericSave):
        def __init__(self, fps=4, dpi=100, bitrate=1800, _type='GIFFileBased', **kwargs):
            super().__init__(**kwargs); assert_requirements()
            if _type == 'GIFPipeBased': writer, extension = animation.ImageMagickFileWriter, '.gif'  # internally calls: matplotlib._get_executable_info("magick")
            elif _type == "GIFFileBased": writer, extension = animation.ImageMagickWriter, '.gif'
            elif _type == 'MPEGFileBased': writer, extension = animation.FFMpegFileWriter, '.mp4'
            elif _type == 'MPEGPipeBased': writer, extension = animation.FFMpegWriter, '.mp4'
            else: raise ValueError("Unknown writer.")
            self.writer = writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = self.save_dir.joinpath(self.save_name + extension)
            assert self.watch_figs, "No figure was sent during instantiation of saver, therefore the writer cannot be setup. Did you mean to use an autosaver?"
            self.writer.setup(fig=self.watch_figs[0], outfile=str(self.fname), dpi=dpi)
        def _save(self, afig, aname, **kwargs): self.writer.grab_frame(**kwargs)
        def finish(self): print('Saving results ...'); self.writer.finish(); print(f"SAVED GIF @", P(self.fname).absolute().as_uri()); return self
    class GIFPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs): super().__init__(*args, _type=self.__class__.__name__, **kwargs)
    class MPEGFileBased(GIFFileBased):
        def __init__(self, *args, **kwargs): super().__init__(*args, _type=self.__class__.__name__, **kwargs)
    class MPEGPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs): super().__init__(*args, _type=self.__class__.__name__, **kwargs)
    class GenericAuto(GenericSave):
        """Parses the data internally, hence requires artist with animate method implemetend. Artist needs to have .fig attribute."""
        save_type = 'auto'
        def __init__(self, plotter_class, data, names_list=None, **kwargs):
            super().__init__(**kwargs); self.saver, self.plotter = None, None; assert_requirements()
            self.plotter_class, self.data, self.names_list, self.kwargs = plotter_class, data, names_list, kwargs
        def animate(self):
            self.plotter = self.plotter_class(**self.kwargs); plt.pause(0.5)  # give time for figures to show up before updating them
            for idx, datum in __import__("tqdm").tqdm(enumerate(self.data)): self.plotter.animate(datum); self.saver.add(names=[self.names_list[idx] if self.names_list is not None else str(idx)])
            self.saver.finish()
    class GIFAuto(GenericAuto):
        def __init__(self, plotter_class, data, interval=500, extension='gif', fps=4, metadata=None, **kwargs):
            super().__init__(plotter_class, data, **kwargs)
            writer = animation.PillowWriter(fps=fps) if extension == '.mp4' else animation.FFMpegWriter(fps=fps, metadata=metadata, bitrate=2500)
            self.plotter = self.plotter_class(**kwargs); plt.pause(self.delay * 0.001)  # give time for figures to show up before updating them
            # noinspection PyTypeChecker
            self.ani = animation.FuncAnimation(fig=self.plotter.fig, func=self.plotter.animate, frames=(i for i in zip(*self.data)), interval=interval, repeat_delay=1500, fargs=None, cache_frame_data=True, save_count=10000)
            self.fname = self.save_dir.joinpath(self.save_name + f".{extension}")
            self.ani.save(filename=self.fname, writer=writer); print(f"SAVED GIF @ ", P(self.fname).absolute().as_uri())
    class PDFAuto(GenericAuto):
        def __init__(self, **kwargs): super().__init__(**kwargs); self.saver = FigureSave.PDF(**kwargs); self.animate()
    class PNGAuto(GenericAuto):
        def __init__(self, **kwargs): super().__init__(**kwargs); self.saver = FigureSave.PNG(**kwargs); self.save_dir = self.saver.save_dir; self.animate(); self.fname = self.saver.fname
    class NullAuto(GenericAuto):
        def __init__(self, **kwargs): super().__init__(**kwargs); self.saver = FigureSave.Null(**kwargs); self.fname = self.saver.fname; self.animate()
    class GIFFileBasedAuto(GenericAuto):
        def __init__(self, plotter_class, data, fps=4, dpi=150, bitrate=2500, _type='GIFFileBasedAuto', **kwargs):
            super().__init__(**kwargs)
            if _type == 'GIFPipeBasedAuto': writer = animation.ImageMagickFileWriter; extension = '.gif'
            elif _type == 'MPEGFileBasedAuto': writer = animation.FFMpegFileWriter; extension = '.mp4'
            elif _type == 'MPEGPipeBasedAuto': writer = animation.FFMpegWriter; extension = '.mp4'
            else: raise ValueError("Unknown writer.")
            self.saver = writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = self.save_dir.joinpath(self.save_name + extension)
            self.data = lambda: (i for i in zip(*data)); self.plotter = plotter_class(*[piece[0] for piece in data], **kwargs); plt.pause(0.5); from tqdm import tqdm
            with self.saver.saving(fig=self.plotter.fig, outfile=self.fname, dpi=dpi):
                for datum in tqdm(self.data()): self.plotter.animate(datum); self.saver.grab_frame(); plt.pause(self.delay * 0.001)
            print(f"SAVED GIF successfully @ {self.fname}")
    class GIFPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs): super().__init__(*args, _type=self.__class__.__name__, **kwargs)
    class MPEGFileBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs): super().__init__(*args, _type=self.__class__.__name__, **kwargs)
    class MPEGPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs): super().__init__(*args, _type=self.__class__.__name__, **kwargs)


class FigureManager:  # use as base class for Artist & Viewers to give it free animation powers.
    def __init__(self, info_loc=None, figpolicy=FigurePolicy.same):
        self.figpolicy = figpolicy
        self.fig = self.ax = self.event = None
        self.cmaps, self.colors, self.mcolors, self.facecolor = Cycle(plt.colormaps()), Cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']), list(mcolors.CSS4_COLORS.keys()), Cycle(list(mcolors.CSS4_COLORS.values()))
        self.cmaps.set_value('viridis')
        self.idx_cycle = Cycle([]); self.pause = None  # animation
        self.help_menu = {'_-=+[{]}\\': {'help': "Adjust Vmin Vmax. Shift + key applies change to all axes \\ toggles auto-brightness ", 'func': self.adjust_brightness},
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
                          'alt+a': {'help': 'Show/Hide annotations', 'func': self.toggle_annotate}}  # IMPORTANT: add the 'alt/ctrl+key' versions of key after the key in the dictionary above, not before, otherwise the naked key version will statisfy the condition `is key in this`? in the parser.
        self.auto_brightness, self.pix_vals = False, False; self.boundaries_flag, self.annot_flag = True, False
        self.info_loc = [0.8, 0.01] if info_loc is None else info_loc
        self.message, self.message_obj, self.cursor = '', None, None
    def show_help(self, event):
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
    def animate(self): pass  # a method of the artist child class that is inheriting from this class to define behaviour when user press next or previous buttons.
    def connect(self): self.fig.canvas.mpl_connect('key_press_event', self.process_key); return self
    def process_key(self, event):
        self.event = event  # useful for debugging.
        for key in self.help_menu.keys():
            if event.key in key: self.help_menu[key]['func'](event); self.update_info_text(self.message); break
        if event.key != 'q': event.canvas.figure.canvas.draw()  # for smooth quit without throwing errors  # don't update if you want to quit.
    def toggle_annotate(self, event):
        self.annot_flag = not self.annot_flag
        if event.inaxes and event.inaxes.images: event.inaxes.images[0].set_picker(True); self.message = f"Annotation flag is toggled to {self.annot_flag}"
    def annotate(self, event, axis=None, data=None):
        self.event = event; e = event.mouseevent; ax = e.inaxes if axis is None else axis
        if not ax: return None
        if not hasattr(ax, 'annot_obj'): ax.annot_obj = ax.annotate("", xy=(0, 0), xytext=(-30, 30), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="w", connectionstyle="arc3"),
                                                                    va="bottom", ha="left", fontsize=10, bbox=dict(boxstyle="round", fc="w"), )
        else: ax.annot_obj.set_visible(self.annot_flag)
        x, y = int(np.round(e.xdata)), int(np.round(e.ydata))
        z = e.inaxes.images[0].get_array()[y, x] if data is None else data[y, x]
        ax.annot_obj.set_text(f'x:{x}\ny:{y}\nvalue:{z:.3f}'); ax.annot_obj.xy = (x, y); self.fig.canvas.draw_idle()
    def save(self, event): _ = event; Save.pickle(path=P.tmpfile(name="figure_manager"), obj=self)
    def replay(self, event): _ = event; self.pause = False; self.idx_cycle.set_index(0); self.message = 'Replaying'; self.animate()
    def pause_func(self, event): _ = event; self.pause = not self.pause; self.message = f'Pause flag is set to {self.pause}'; self.animate()
    def previous(self, event): _ = event; self.idx_cycle.previous(); self.message = f'Previous {self.idx_cycle}'; self.animate()
    def next(self, event): _ = event; self.idx_cycle.next(); self.message = f'Next {self.idx_cycle}'; self.animate()
    def text_info(self, event): _ = event; self.message = ''
    def change_facecolor(self, event): self.fig.set_facecolor(self.facecolor.next() if event.key == '>' else self.facecolor.previous()); self.message = f"Figure facecolor was set to {self.mcolors[self.facecolor.get_index()]}"
    def adjust_brightness(self, event):
        ax, message = event.inaxes, "None"
        if ax is None or not ax.images: return None
        if event.key == '\\':
            self.auto_brightness = not self.auto_brightness; message = f"Auto-brightness flag is toggled to {self.auto_brightness}"
            if self.auto_brightness: im = self.ax.images[0]; im.norm.autoscale(im.get_array())  # changes to all ims take place in animate as in ImShow and Nifti methods animate.
        vmin, vmax = ax.images[0].get_clim()
        if event.key in '-_': message = 'increase vmin'; vmin += 1
        elif event.key in '[{': message = 'decrease vmin'; vmin -= 1
        elif event.key in '=+': message = 'increase vmax'; vmax += 1
        elif event.key in ']}': message = 'decrease vmax'; vmax -= 1
        self.message = message + '  ' + str(round(vmin, 1)) + '  ' + str(round(vmax, 1))
        if event.key in '_+}{': [ax.images[0].set_clim((vmin, vmax)) for ax in self.fig.axes if ax.images]
        else: ax.images[0].set_clim((vmin, vmax)) if ax.images else None
    def change_cmap(self, event):
        if ax := event.inaxes is not None:
            cmap = self.cmaps.next() if event.key in 'tT' else self.cmaps.previous()
            [[im.set_cmap(cmap) for im in ax.images] for ax in self.fig.axe] if event.key in 'TY'else [im.set_cmap(cmap) for im in ax.images]
            self.message = f"Color map changed to {ax.images[0].cmap.name}"
    def show_pix_val(self, event):
        if (ax := event.inaxes) is not None:
            self.pix_vals = not self.pix_vals; self.message = f"Pixel values flag set to {self.pix_vals}"
            if self.pix_vals: self.show_pixels_values(ax)
            else:
                while len(ax.texts) > 0: [text.remove() for text in ax.texts]
    def show_cursor(self, event):
        if not (ax := event.inaxes): return None  # don't do this if c was pressed outside an axis.
        if hasattr(ax, 'cursor_'):  # is this the first time?
            if ax.cursor_ is None: ax.cursor_ = widgets.Cursor(ax=ax, vertOn=True, horizOn=True, color='red', lw=1.0)
            else: ax.cursor_ = None  # toggle the cursor.
            self.message = f'Cursor flag set to {bool(ax.cursor_)}'
        else: ax.cursor_ = None; self.show_cursor(event)  # first call
    def show_ticks(self, event):
        self.boundaries_flag = not self.boundaries_flag
        if event.key == 'a' and (axis := event.inaxes): self.toggle_ticks(axis); self.message = f"Boundaries flag set to {self.boundaries_flag} in {axis}"
        else: [self.toggle_ticks(ax) for ax in self.ax]
    # ====================== class methods ===============================
    def get_fig(self, figname='', suffix=None, **kwargs): return FigureManager.get_fig_static(self.figpolicy, figname, suffix, **kwargs)
    def update_info_text(self, message): self.message_obj.remove() if self.message_obj else None; self.message_obj = self.fig.text(*self.info_loc, message, fontsize=8)
    def maximize_fig(self): _ = self; plt.get_current_fig_manager().full_screen_toggle()  # TODO not working appropriately ImShow.test() # The command required is backend-dependent and also OS dependent. Doesn't work if figure is not shown yet.
    def clear_axes(self): [ax.cla() for ax in self.ax]
    def transperent_fig(self): self.fig.canvas.manager.window.attributes("-transparentcolor", "white")
    def close(self): plt.close(self.fig)
    # ====================== axis helpers ========================
    @staticmethod
    def grid(ax, factor=5, x_or_y='both', color='gray', alpha1=0.5, alpha2=0.25):
        if type(ax) in {list, List, np.ndarray}: [FigureManager.grid(an_ax, factor=factor, x_or_y=x_or_y, color=color, alpha1=alpha1, alpha2=alpha2) for an_ax in ax]  # Turning on major grid for both axes.
        ax.grid(which='major', axis='x', color='gray', linewidth=0.5, alpha=alpha1); ax.grid(which='major', axis='y', color='gray', linewidth=0.5, alpha=alpha1)
        if x_or_y in {'both', 'x'}: xt = ax.get_xticks(); ax.xaxis.set_minor_locator(plt.MultipleLocator((xt[1] - xt[0]) / factor)); ax.grid(which='minor', axis='x', color=color, linewidth=0.5, alpha=alpha2)
        if x_or_y in {'both', 'y'}: yt = ax.get_yticks(); ax.yaxis.set_minor_locator(plt.MultipleLocator((yt[1] - yt[0]) / factor)); ax.grid(which='minor', axis='y', color=color, linewidth=0.5, alpha=alpha2)
    @staticmethod
    def set_ax_size(ax, w, h, units='inches'):
        l, r, t, b, _ = ax.figure.subplotpars.left, ax.figure.subplotpars.right, ax.figure.subplotpars.top, ax.figure.subplotpars.bottom, units
        ax.figure.set_size_inches(float(w) / (r - l), float(h) / (t - b))
    @staticmethod
    def get_ax_size(ax, units="inches"):
        w, h = ax.figure.get_size_inches() if units == "inches" else ax.figure.get_size_pixels()
        width, height = ax.figure.subplotpars.right - ax.figure.subplotpars.left, ax.figure.subplotpars.top - ax.figure.subplotpars.bottom
        return w * width, h * height
    @staticmethod
    def toggle_ticks(an_ax, state=None): [line.set_visible(not line.get_visible() if state is None else state) for line in an_ax.get_yticklines() + an_ax.get_xticklines() + an_ax.get_xticklabels() + an_ax.get_yticklabels()]
    @staticmethod
    def set_ax_to_real_life_size(ax, inch_per_unit=1 / 25.4): FigureManager.set_ax_size(ax, (ax.get_xlim()[1] - ax.get_xlim()[0]) * inch_per_unit, (ax.get_ylim()[1] - ax.get_ylim()[0]) * inch_per_unit)
    @staticmethod
    def show_pixels_values(ax):
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        if ymin > ymax: ymin, ymax = ymax, ymin  # default imshow settings
        [ax.text(i, j, np.round(label).__int__(), ha='center', va='center', size=8) for (j, i), label in np.ndenumerate(ax.images[0].get_array()) if (xmin < i < xmax) and (ymin < j < ymax)]
    # ============================ matplotlib setup ==============================
    @staticmethod
    def get_nrows_ncols(num_plots, nrows=None, ncols=None):
        if not nrows and not ncols:
            nrows, ncols = int(np.floor(np.sqrt(num_plots))), int(np.ceil(np.sqrt(num_plots)))
            while nrows * ncols < num_plots: ncols += 1
        elif not ncols and nrows: ncols = int(np.ceil(num_plots / nrows))
        elif not nrows and ncols: nrows = int(np.ceil(num_plots / ncols))
        return nrows, ncols
    @staticmethod
    def findobj(figname, obj_name): return (plt.figure(num=figname) if type(figname) is str else figname).findobj(lambda x: x.get_label() == obj_name)
    @staticmethod
    def get_fig_static(figpolicy, figname='', suffix=None, **kwargs):
        exist = True if figname in plt.get_figlabels() else False
        if figpolicy is FigurePolicy.same: return plt.figure(num=figname, **kwargs)
        elif figpolicy is FigurePolicy.add_new: return plt.figure(num=(timestamp(name=figname) if suffix is None else figname + suffix) if exist else figname, **kwargs)
        elif figpolicy is FigurePolicy.close_create_new: plt.close(figname) if exist else None; return plt.figure(num=figname, **kwargs)
    @staticmethod
    def try_figure_size():
        fig, ax = plt.subplots()
        y = np.sin(x := np.arange(0, 100, 0.01)) * 100
        ax.plot(x, y); ax.axis("square"); ax.set_xlim(0, 100); ax.set_ylim(-100, 100)
        FigureManager.set_ax_to_real_life_size(ax); fig.savefig(P.tmp() / "trial.png", dpi=250)
    @staticmethod
    def write(txt, name="text", size=8, **kwargs):
        FigureManager.maximize_fig(fig := plt.figure(figsize=(11.69, 8.27), num=name))
        fig.clf(); fig.text(0.5, 0.5, txt, transform=fig.transFigure, size=size, ha="center", va='center', **kwargs); return fig
    @staticmethod
    def activate_latex(size=20):
        plt.rc('xtick', labelsize=size); plt.rc('ytick', labelsize=size)
        plt.rc('axes', titlesize=size); plt.rc('legend', fontsize=size / 1.5)  # rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True; plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    @staticmethod
    def set_linestyles_and_markers_and_colors(test=False):
        from cycler import cycler; from matplotlib import lines
        markers = list(lines.lineMarkers.keys())[:-4]  # ignore the None
        linestyles = (list(lines.lineStyles.keys())[:-3] * 10)[:len(markers)]
        colors = (plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10)[:len(markers)]
        default_cycler = (cycler(linestyle=linestyles) + cycler(marker=markers) + cycler(color=colors))
        plt.rc('axes', prop_cycle=default_cycler)
        if test: [plt.plot(aq + idx * 2) for idx, aq in enumerate(np.random.randn(10, 10))]


class VisibilityViewer(FigureManager):  # This is used for browsing purpose, as opposed to saving.
    artist = ['internal', 'external'][1]  # How is the data visualized? You need artist. The artist can either be internal, as in ImShow or passed externally (especially non image data)
    parser = ['internal', 'external'][1]  # Data parsing: internal for loop to go through all the dataset passed. # Allows manual control over parsing. external for loop. It should have add method. # Manual control only takes place after the external loop is over. #TODO parallelize this.
    stream = ['clear', 'accumulate', 'update'][1]  # Streaming (Refresh mechanism): * Clear the axis. (slowest, but easy on memory) * accumulate, using visibility to hide previous axes. (Fastest but memory intensive)  * The artist has an update method. (best)
    """ This class works on hiding axes shown on a plot, so that a new plot can be drawn. Once the entire loop is finished, you can browse through the plots with the keyboard Animation linked to `animate`"""
    def __init__(self, fig):  # Downside: slow if number of axes, lines, texts, etc. are too large. In that case, it is better to avoid this viewer and plot on the fly during animation.
        super().__init__(); self.objs_repo = []  # list of lists of axes and texts.
        self.fig = fig; self.connect(); self.idx_cycle = Cycle([])
    def add(self, objs): self.idx_cycle.expand(); self.objs_repo.append(objs); [obj.set_visible(False) for obj in (self.objs_repo[-2] if len(self.objs_repo) > 1 else [])]; print(f"VViewer added plot number {self.idx_cycle.get_index()}", end='\r')
    def animate(self): [ax.set_visible(False) for ax in self.objs_repo[self.idx_cycle.prev_index]]; [ax.set_visible(True) for ax in self.objs_repo[self.idx_cycle.get_index()]]; self.fig.canvas.draw()


class Artist(FigureManager):  # This object knows how to draw a figure from curve-type data.
    def __init__(self, ax=None, figname='Graph', title='', label='curve', style='seaborn', figpolicy=FigurePolicy.add_new, figsize=(14, 8)):
        super().__init__(figpolicy=figpolicy)
        self.style, self.title = style, title; self.line = self.cursor = self.check_b = None
        if ax is None:  # create a figure
            with plt.style.context(style=self.style): self.fig = self.get_fig(figname, figsize=figsize); self.ax = self.fig.subplots()
        else: self.ax = ax; self.fig = ax.figure  # use the passed axis
        self.visibility_ax, self.txt, self.label = [0.01, 0.05, 0.2, 0.15], [], label
    def plot(self, *args, legends=None, title=None, **kwargs): self.line = self.ax.plot(*args, **kwargs); self.ax.legend(legends or []); self.ax.set_title(title) if title is not None else None; self.ax.grid('on')
    def plot_dict(self, adict, title='', xlabel='', ylabel=''): [self.plot(val, label=key) for key, val in adict.items()]; self.ax.legend(); self.ax.set_title(title); self.ax.set_xlabel(xlabel); self.ax.set_ylabel(ylabel); return self
    def plot_twin(self, c1, c2, x=None, l1='', l2='', ax=None): ax = ax or self.ax; twin_ax = ax.twinx(); line1 = ax.plot(x or range(len(c1)), c1, color="blue", label=l1)[0]; line2 = twin_ax.plot(c2, color="red", label=l2)[0]; twin_ax.legend([line1, line2], [l1, l2]); ax.set_ylabel(l1); twin_ax.set_ylabel(l2); plt.show()
    def suptitle(self, title): self.txt = [self.fig.text(0.5, 0.98, title, ha='center', va='center', size=9)]
    def clear(self): self.fig.clf()  # objects that use this class will demand that the artist expose this method, in addition to .plot()
    def axes(self): return [self.ax]  # objects that use this class will demand that the artist expose this method, in addition to .plot()
    def visibility(self):
        from matplotlib.widgets import CheckButtons; self.fig.subplots_adjust(left=0.3); self.visibility_ax[-1] = 0.05 * len(self.ax.lines)
        rax = self.fig.add_axes(self.visibility_ax)
        labels, visibility = [str(line.get_label()) for line in self.ax.lines], [line.get_visible() for line in self.ax.lines]
        self.check_b = CheckButtons(rax, labels, visibility)
        def func(label): index = labels.index(label); self.ax.lines[index].set_visible(not self.ax.lines[index].get_visible()); self.fig.canvas.draw()
        self.check_b.on_clicked(func)
    @staticmethod
    def styler(plot_gen):
        for astyle in plt.style.available:
            with plt.style.context(style=astyle): plot_gen(); plt.title(astyle); plt.pause(1); plt.cla()


class VisibilityViewerAuto(VisibilityViewer):
    artist = ['internal', 'external'][1]
    parser = ['internal', 'external'][0]
    stream = ['clear', 'accumulate', 'update'][0:2]
    def __init__(self, data=None, artist=None, stream='clear', save_type=FigureSave.Null, save_dir=None, save_name=None, delay=1,
                 titles=None, legends=None, x_labels=None, pause=True, **kwargs):
        """data: tensor of form  NumInputsForAnimation x ArgsPerPlot (to be unstarred) x Input (possible points x signals)
        stream: ensure that behaviour of artist is consistent with stream. When `cccumulate`, artist should create new axes whenever plot is called."""
        self.data = data; self.artist = artist or Artist()
        self.legends = [f"Plot {i}" for i in range(len(self.data))] if legends is None else legends; self.titles = titles if titles is not None else np.arange(len(self.data))
        super().__init__(fig=self.artist.fig); _ = kwargs
        self.saver = save_type(watch_figs=[self.artist.fig], save_dir=save_dir, save_name=save_name, delay=delay, fps=1000 / delay)
        self.index_max, self.pause, self.stream, self.lables = len(self.data), pause, stream, x_labels
    test = staticmethod(lambda: VisibilityViewerAuto(data=np.random.randn(10, 1, 10, 3)))
    def animate(self):
        for i in range(0, self.index_max):
            self.artist.plot(*self.data[i], title=self.titles[i], legends=self.legends)  # replot the new data point on a new axis.
            super().add(self.artist.axes()) if self.stream == 'accumulate' else self.artist.clear()
            self.saver.add()
            if self.pause: break
            else: self.idx_cycle.set_index(i)
        return self.saver.finish()  # arrived at last image and not in manual mode


class ImShow(FigureManager):
    artist = ['internal', 'external'][0]
    parser = ['internal', 'external'][0]
    stream = ['clear', 'accumulate', 'update'][2]
    def __init__(self, img_tensor, sup_titles=None, sub_labels=None, save_type=FigureSave.Null, save_name=None, save_dir=None, save_kwargs=None,
                 subplots_adjust=None, gridspec=None, tight=True, info_loc=None, nrows=None, ncols=None, ax=None,
                 figsize=None, figname='im_show', auto_brightness=True, delay=200, pause=False, **kwargs):
        """
        :param img_tensor: size N x M x W x H [x C]  # M used spatially, N for animation.
        :param sup_titles: Titles for frames (N)
        :param sub_labels: M x N. If shape sent is M
        """
        n, m = len(img_tensor), len(img_tensor[0]); self.m, self.n = m, n; super(ImShow, self).__init__(info_loc=info_loc)
        nrows, ncols = self.get_nrows_ncols(m, nrows, ncols)
        self.img_tensor, self.sub_labels, self.sup_titles = img_tensor, sub_labels if sub_labels is not None else [[f"{i}-{j}" for j in range(m)] for i in range(n)], sup_titles if sup_titles is not None else np.arange(n)
        self.pause, self.kwargs, self.delay, self.auto_brightness = pause, kwargs, delay, auto_brightness
        self.fname = self.event = None; self.ims = []  # container for images.
        self.cmaps = Cycle(plt.colormaps()); self.cmaps.set_value('viridis')
        if ax is None:
            self.fig = self.get_fig(figname=figname, figsize=(14, 9) if figsize is None else figsize, facecolor='white'); self.maximize_fig() if figsize is None else None
            if gridspec is not None: gs = self.fig.add_gridspec(gridspec[0]); self.ax = [self.fig.add_subplot(gs[ags[0], ags[1]]) for ags in gs[1:]]
            else: self.ax = self.fig.subplots(nrows=nrows, ncols=ncols)
        else: self.ax = ax; self.fig = ax[0].figure if type(ax) is list else ax.figure
        self.connect(); self.fig.canvas.mpl_connect("pick_event", self.annotate)
        self.fig.tight_layout() if tight else None; self.fig.subplots_adjust(**subplots_adjust) if subplots_adjust is not None else None
        self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name, delay=delay, fps=1000 / delay, **({} if save_kwargs is None else save_kwargs))
        self.ax = [self.ax] if nrows == 1 and ncols == 1 else self.ax.ravel()  # make a list out of it or # make a 1D  list out of a 2D array.
        [self.toggle_ticks(an_ax, state=False) for an_ax in self.ax]; self.idx_cycle = Cycle(max_idx=len(self.img_tensor))  # self.animate()
    def animate(self):
        for i in range(self.idx_cycle.get_index(), self.n):
            for j, (an_image, a_label, an_ax) in enumerate(zip(self.img_tensor[i], self.sub_labels[i], self.ax)):  # with zipping, the shortest of the three, will stop the loop.
                if i == 0 and self.ims.__len__() < self.m: self.ims.append(an_ax.imshow(an_image, animated=True, **self.kwargs))
                else: self.ims[j].set_data(an_image)
                if self.auto_brightness: self.ims[j].norm.autoscale(an_image)
                an_ax.set_xlabel(f'{a_label}')
            self.fig.suptitle(self.sup_titles[i], fontsize=8); self.saver.add(names=[self.sup_titles[i]])
            if self.pause: break
            else: self.idx_cycle.set_index(i)
        if self.idx_cycle.get_index() == self.n - 1 and not self.pause: self.fname = self.saver.finish()  # arrived at last image and not in manual mode
    @staticmethod
    def try_cmaps(im, nrows=3, ncols=7, **kwargs): _ = ImShow(*np.array_split([plt.get_cmap(style)(im) for style in plt.colormaps()], nrows * ncols), nrows=nrows, ncols=ncols, sub_labels=np.array_split(plt.colormaps(), nrows * ncols), **kwargs); return [plt.get_cmap(style)(im) for style in plt.colormaps()]
    def annotate(self, event, axis=None, data=None): [super().annotate(event, axis=ax, data=ax.images[0].get_array()) for ax in self.ax]
    from_img_paths = staticmethod(lambda paths, **kwargs: ImShow(List(paths).apply(plt.imread), sub_labels=List(paths).apply(lambda x: P(x).stem), **kwargs))
    from_complex = staticmethod(lambda data, pause=True, **kwargs: ImShow(data.real, data.imag, np.angle(data), abs(data), labels=['Real Part', 'Imaginary Part', 'Angle in Radians', 'Absolute Value'], pause=pause, **kwargs))
    test = staticmethod(lambda: ImShow(np.random.rand(12, 10, 80, 120, 3)))  # https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
    resize = staticmethod(lambda path, m, n: plt.imsave(path, install_n_import(package="skimage", name="scikit-image").transform.resize(plt.imread(path), (m, n), anti_aliasing=True)))


if __name__ == '__main__':
    pass
