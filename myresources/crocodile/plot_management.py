
import enum
import matplotlib.pyplot as plt
from crocodile.core import np, List, timestamp, os, Save
from crocodile.file_management import P
import typing
import pandas as pd
from crocodile.meta import Cycle


class FigurePolicy(enum.Enum):
    close_create_new = 'Close the previous figure that has the same figname and create a new fresh one'
    add_new = 'Create a new figure with same name but with added suffix'
    same = 'Grab the figure of the same name'


class FigureManager:
    """
    Handles figures of matplotlib.
    """

    def __init__(self, info_loc=None, figpolicy=FigurePolicy.same):
        self.figpolicy = figpolicy
        self.fig = self.ax = self.event = None
        self.cmaps = Cycle(plt.colormaps())
        import matplotlib.colors as mcolors
        self.mcolors = list(mcolors.CSS4_COLORS.keys())
        self.facecolor = Cycle(list(mcolors.CSS4_COLORS.values()))
        self.colors = Cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        self.cmaps.set('viridis')
        self.index = self.pause = self.index_max = None
        self.auto_brightness = False
        self.info_loc = [0.8, 0.01] if info_loc is None else info_loc
        self.pix_vals = False
        self.help_menu = {'_-=+[{]}\\': {'help': "Adjust Vmin Vmax. Shift + key applies change to all axes. \\ "
                                                 "toggles auto-brightness ", 'func': self.adjust_brightness},
                          "/": {'help': 'Show/Hide info text', 'func': self.text_info},
                          "h": {'help': 'Show/Hide help menu', 'func': self.show_help},
                          "tyTY": {'help': 'Change color map', 'func': self.change_cmap},
                          '<>': {'help': 'Change figure face color', 'func': self.change_facecolor},
                          "v": {'help': 'Show/Hide pixel values (Toggle)', 'func': self.show_pix_val},
                          'P': {'help': 'Pause/Proceed (Toggle)', 'func': self.pause_func},
                          'r': {'help': 'Replay', 'func': self.replay},
                          '1': {'help': 'Previous Image', 'func': self.previous},
                          '2': {'help': 'Next Image', 'func': self.next},
                          'S': {'help': 'Save Object', 'func': self.save},
                          'c': {'help': 'Show/Hide cursor', 'func': self.show_cursor},
                          'aA': {'help': 'Show/Hide ticks and their labels', 'func': self.show_ticks},
                          'alt+a': {'help': 'Show/Hide annotations', 'func': self.toggle_annotate}}
        # IMPORTANT: add the 'alt/ctrl+key' versions of key after the key in the dictionary above, not before.
        # Otherwise the naked key version will statisfy the condition `is key in this`? in the parser.
        self.message = ''
        self.message_obj = self.cursor = None
        self.annot_flag = False  # one flag for all axes?
        self.boundaries_flag = True

    @staticmethod
    def grid(ax, factor=5, x_or_y='both', color='gray', alpha1=0.5, alpha2=0.25):
        """
        :param ax: Axis object from matplotlib
        :param factor: number of major divisions.
        :param x_or_y: which axis to grid.
        :param color: grid color
        :param alpha1: transparancy for x axis grid.
        :param alpha2: transparancy for y axis grid.
        :return:
        """
        if type(ax) in {list, List, np.ndarray}:
            for an_ax in ax:
                FigureManager.grid(an_ax, factor=factor, x_or_y=x_or_y, color=color, alpha1=alpha1, alpha2=alpha2)
            return None

        # Turning on major grid for both axes.
        ax.grid(which='major', axis='x', color='gray', linewidth=0.5, alpha=alpha1)
        ax.grid(which='major', axis='y', color='gray', linewidth=0.5, alpha=alpha1)
        if x_or_y in {'both', 'x'}:
            xt = ax.get_xticks()  # major ticks
            steps = (xt[1] - xt[0]) / factor
            ax.xaxis.set_minor_locator(plt.MultipleLocator(steps))
            ax.grid(which='minor', axis='x', color=color, linewidth=0.5, alpha=alpha2)
        if x_or_y in {'both', 'y'}:
            yt = ax.get_yticks()  # major ticks
            steps = (yt[1] - yt[0]) / factor
            ax.yaxis.set_minor_locator(plt.MultipleLocator(steps))
            ax.grid(which='minor', axis='y', color=color, linewidth=0.5, alpha=alpha2)

    def maximize_fig(self):
        """The command required is backend-dependent and also OS dependent."""
        _ = self
        # plt.get_current_fig_manager().window.state('zoom')
        plt.get_current_fig_manager().full_screen_toggle()

    def toggle_annotate(self, event):
        self.annot_flag = not self.annot_flag
        if event.inaxes:
            if event.inaxes.images:
                event.inaxes.images[0].set_picker(True)
                self.message = f"Annotation flag is toggled to {self.annot_flag}"
        if not self.annot_flag:  # if it is off
            pass  # hide all annotations

    def annotate(self, event, axis=None, data=None):
        self.event = event
        e = event.mouseevent
        if axis is None:
            ax = e.inaxes
        else:
            ax = axis
        if ax:
            if not hasattr(ax, 'annot_obj'):  # first time
                ax.annot_obj = ax.annotate("", xy=(0, 0), xytext=(-30, 30),
                                           textcoords="offset points",
                                           arrowprops=dict(arrowstyle="->", color="w", connectionstyle="arc3"),
                                           va="bottom", ha="left", fontsize=10,
                                           bbox=dict(boxstyle="round", fc="w"), )
            else:
                ax.annot_obj.set_visible(self.annot_flag)

            x, y = int(np.round(e.xdata)), int(np.round(e.ydata))
            if data is None:
                z = e.inaxes.images[0].get_array()[y, x]
            else:
                z = data[y, x]
            ax.annot_obj.set_text(f'x:{x}\ny:{y}\nvalue:{z:.3f}')
            ax.annot_obj.xy = (x, y)
            self.fig.canvas.draw_idle()

    def save(self, event):
        _ = event
        # noinspection PyTypeChecker
        Save.pickle('.', obj=self)

    def replay(self, event):
        _ = event
        self.pause = False
        self.index = 0
        self.message = 'Replaying'
        self.animate()

    def pause_func(self, event):
        _ = event
        self.pause = not self.pause
        self.message = f'Pause flag is set to {self.pause}'
        self.animate()

    def previous(self, event):
        _ = event
        self.index = self.index - 1 if self.index > 0 else self.index_max - 1
        self.message = f'Previous {self.index}'
        self.animate()

    def next(self, event):
        _ = event
        self.index = self.index + 1 if self.index < self.index_max - 1 else 0
        self.message = f'Next {self.index}'
        self.animate()

    def animate(self):
        pass  # a method of the artist child class that is inheriting from this class

    def text_info(self, event):
        _ = event
        self.message = ''

    def show_help(self, event):
        _ = event
        default_plt = {"q ": {'help': "Quit Figure."},
                       "Ll": {'help': "change x/y scale to log and back to linear (toggle)"},
                       "Gg": {'help': "Turn on and off x and y grid respectively."},
                       "s ": {'help': "Save Figure"},
                       "f ": {'help': "Toggle Full screen"},
                       "p ": {'help': "Select / Deselect Pan"}}
        figs = plt.get_figlabels()
        if "Keyboard shortcuts" in figs:
            plt.close("Keyboard shortcuts")  # toggle
        else:
            fig = plt.figure(num="Keyboard shortcuts")
            for i, key in enumerate(self.help_menu.keys()):
                fig.text(0.1, 1 - 0.05 * (i + 1), f"{key:30s} {self.help_menu[key]['help']}")
            print(pd.DataFrame([[val['help'], key] for key, val in self.help_menu.items()], columns=['Action', 'Key']))
            print(f"\nDefault plt Keys:\n")
            print(pd.DataFrame([[val['help'], key] for key, val in default_plt.items()], columns=['Action', 'Key']))

    def adjust_brightness(self, event):
        ax = event.inaxes
        if ax is not None and ax.images:
            message = 'None'
            if event.key == '\\':
                self.auto_brightness = not self.auto_brightness
                message = f"Auto-brightness flag is set to {self.auto_brightness}"
                if self.auto_brightness:  # this change is only for the current image.
                    im = self.ax.images[0]
                    im.norm.autoscale(im.get_array())
                    # changes to all ims take place in animate as in ImShow and Nifti methods animate.
            vmin, vmax = ax.images[0].get_clim()
            if event.key in '-_':
                message = 'increase vmin'
                vmin += 1
            elif event.key in '[{':
                message = 'decrease vmin'
                vmin -= 1
            elif event.key in '=+':
                message = 'increase vmax'
                vmax += 1
            elif event.key in ']}':
                message = 'decrease vmax'
                vmax -= 1
            self.message = message + '  ' + str(round(vmin, 1)) + '  ' + str(round(vmax, 1))
            if event.key in '_+}{':
                for ax in self.fig.axes:
                    if ax.images:
                        ax.images[0].set_clim((vmin, vmax))
            else:
                if ax.images:
                    ax.images[0].set_clim((vmin, vmax))

    def change_cmap(self, event):
        ax = event.inaxes
        if ax is not None:
            cmap = self.cmaps.next() if event.key in 'tT' else self.cmaps.previous()
            if event.key in 'TY':
                for ax in self.fig.axes:
                    for im in ax.images:
                        im.set_cmap(cmap)
            else:
                for im in ax.images:
                    im.set_cmap(cmap)
            self.message = f"Color map changed to {ax.images[0].cmap.name}"

    def change_facecolor(self, event):
        color = self.facecolor.next() if event.key == '>' else self.facecolor.previous()
        self.fig.set_facecolor(color)
        self.message = f"Figure facecolor was set to {self.mcolors[self.facecolor.get_index()]}"

    def show_pix_val(self, event):
        ax = event.inaxes
        if ax is not None:
            self.pix_vals = not self.pix_vals  # toggle
            self.message = f"Pixel values flag set to {self.pix_vals}"
            if self.pix_vals:
                self.show_pixels_values(ax)
            else:
                while len(ax.texts) > 0:
                    for text in ax.texts:
                        text.remove()

    def process_key(self, event):
        self.event = event  # useful for debugging.
        for key in self.help_menu.keys():
            if event.key in key:
                self.help_menu[key]['func'](event)
                break
        self.update_info_text(self.message)
        if event.key != 'q':  # for smooth quit without throwing errors
            fig = event.canvas.figure  # don't update if you want to quit.
            fig.canvas.draw()

    def update_info_text(self, message):
        if self.message_obj:
            self.message_obj.remove()
        self.message_obj = self.fig.text(*self.info_loc, message, fontsize=8)

    @staticmethod
    def get_nrows_ncols(num_plots, nrows=None, ncols=None):
        if not nrows and not ncols:
            nrows = int(np.floor(np.sqrt(num_plots)))
            ncols = int(np.ceil(np.sqrt(num_plots)))
            while nrows * ncols < num_plots:
                ncols += 1
        elif not ncols and nrows:
            ncols = int(np.ceil(num_plots / nrows))
        elif not nrows and ncols:
            nrows = int(np.ceil(num_plots / ncols))
        else:
            pass
        return nrows, ncols

    def show_cursor(self, event):
        ax = event.inaxes
        if ax:  # don't do this if c was pressed outside an axis.
            if hasattr(ax, 'cursor_'):  # is this the first time?
                if ax.cursor_ is None:
                    from matplotlib import widgets
                    ax.cursor_ = widgets.Cursor(ax=ax, vertOn=True, horizOn=True, color='red', lw=1.0)
                else:  # toggle the cursor.
                    ax.cursor_ = None
                self.message = f'Cursor flag set to {bool(ax.cursor_)}'
            else:  # first call
                ax.cursor_ = None
                self.show_cursor(event)

    def show_ticks(self, event):
        self.boundaries_flag = not self.boundaries_flag
        axis = event.inaxes
        if event.key == 'a':
            if axis:
                # event.inaxes.axis(['off', 'on'][self.boundaries_flag])
                self.toggle_ticks(axis)
                self.message = f"Boundaries flag set to {self.boundaries_flag} in {axis}"

        else:
            for ax in self.ax:
                # ax.axis(['off', 'on'][self.boundaries_flag])
                self.toggle_ticks(ax)

    @staticmethod
    def toggle_ticks(an_ax, state=None):
        for line in an_ax.get_yticklines():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_xticklines():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_xticklabels():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_yticklabels():
            line.set_visible(not line.get_visible() if state is None else state)

    def clear_axes(self):
        for ax in self.ax:
            ax.cla()

    @staticmethod
    def show_pixels_values(ax):
        im = ax.images[0].get_array()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if ymin > ymax:  # default imshow settings
            ymin, ymax = ymax, ymin
        for (j, i), label in np.ndenumerate(im):
            if (xmin < i < xmax) and (ymin < j < ymax):
                ax.text(i, j, np.round(label).__int__(), ha='center', va='center', size=8)

    @staticmethod
    def update(figname, obj_name, data=None):
        """Fastest update ever. But, you need access to label name.
        Using this function external to the plotter. But inside the plotter you need to define labels to objects
        The other alternative is to do the update inside the plotter, but it will become very verbose.

        :param figname:
        :param obj_name:
        :param data:
        :return:
        """
        obj = FigureManager.findobj(figname, obj_name)
        if data is not None:
            obj.set_data(data)
            # update scale:
            # obj.axes.relim()
            # obj.axes.autoscale()

    @staticmethod
    def findobj(figname, obj_name):
        if type(figname) is str:
            fig = plt.figure(num=figname)
        else:
            fig = figname
        search_results = fig.findobj(lambda x: x.get_label() == obj_name)
        if len(search_results) > 0:  # list of length 1, 2 ...
            search_results = search_results[0]  # the first one is good enough.
        return search_results

    def get_fig(self, figname='', suffix=None, **kwargs):
        return FigureManager.get_fig_static(self.figpolicy, figname, suffix, **kwargs)

    @staticmethod
    def get_fig_static(figpolicy, figname='', suffix=None, **kwargs):
        """
        :param figpolicy:
        :param figname:
        :param suffix: only relevant if figpolicy is add_new
        :param kwargs:
        :return:
        """
        fig = None
        exist = True if figname in plt.get_figlabels() else False
        if figpolicy is FigurePolicy.same:
            fig = plt.figure(num=figname, **kwargs)
        elif figpolicy is FigurePolicy.add_new:
            if exist:
                new_name = timestamp(name=figname) if suffix is None else figname + suffix
            else:
                new_name = figname
            fig = plt.figure(num=new_name, **kwargs)
        elif figpolicy is FigurePolicy.close_create_new:
            if exist:
                plt.close(figname)
            fig = plt.figure(num=figname, **kwargs)
        return fig

    def transperent_fig(self):
        self.fig.canvas.manager.window.attributes("-transparentcolor", "white")

    @staticmethod
    def set_ax_size(ax, w, h):
        """ w, h: width, height in inches """
        left = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - left)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)

    @staticmethod
    def get_ax_size(ax):  # returns axis size in inches.
        w, h = ax.figure.get_size_inches()
        width = ax.figure.subplotpars.right - ax.figure.subplotpars.left
        height = ax.figure.subplotpars.top - ax.figure.subplotpars.bottom
        # width, height = ax.figbox.extents[2:] - ax.figbox.extents[:2]
        return w * width, h * height

    @staticmethod
    def set_ax_to_real_life_size(ax, inch_per_unit=1 / 25.4):
        limit_x = ax.get_xlim()[1] - ax.get_xlim()[0]
        limit_y = ax.get_ylim()[1] - ax.get_ylim()[0]
        FigureManager.set_ax_size(ax, limit_x * inch_per_unit, limit_y * inch_per_unit)

    @staticmethod
    def try_figure_size():
        fig, ax = plt.subplots()
        x = np.arange(0, 100, 0.01)
        y = np.sin(x) * 100
        ax.plot(x, y)
        ax.axis("square")
        ax.set_xlim(0, 100)
        ax.set_ylim(-100, 100)
        FigureManager.set_ax_to_real_life_size(ax)
        fig.savefig(P.tmp() / "trial.png", dpi=250)

    @staticmethod
    def write(txt, name="text", size=8, **kwargs):
        fig = plt.figure(figsize=(11.69, 8.27), num=name)
        FigureManager.maximize_fig(fig)
        fig.clf()
        fig.text(0.5, 0.5, txt, transform=fig.transFigure, size=size, ha="center", va='center', **kwargs)
        return fig

    @staticmethod
    def activate_latex(size=20):
        """Setting up matplotlib"""
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('axes', labelsize=size)
        plt.rc('axes', titlesize=size)
        plt.rc('legend', fontsize=size / 1.5)
        # rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    @staticmethod
    def set_linestyles_and_markers_and_colors(test=False):
        from cycler import cycler
        from matplotlib import lines
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = list(lines.lineMarkers.keys())[:-4]  # ignore the None
        linestyles = list(lines.lineStyles.keys())[:-3]  # ignore the Nones
        linestyles = (linestyles * 10)[:len(markers)]
        colors = (colors * 10)[:len(markers)]
        default_cycler = (cycler(linestyle=linestyles) + cycler(marker=markers) + cycler(color=colors))
        plt.rc('axes', prop_cycle=default_cycler)
        if test:
            temp = np.random.randn(10, 10)
            for idx, aq in enumerate(temp):
                plt.plot(aq + idx * 2)

    def close(self):
        plt.close(self.fig)


class SaveType:
    class GenericSave:
        """ You can either pass the figures to be tracked or, pass them dynamically at add method, or,
        add method will capture every figure and axis

        """
        stream = ['clear', 'accumulate', 'update'][0]

        def __init__(self, save_dir=None, save_name=None, watch_figs=None, max_calls=2000, delay=100, **kwargs):
            self.delay = delay
            self.watch_figs = watch_figs
            if watch_figs:
                assert type(watch_figs) is list, "This should be a list"
                if type(watch_figs[0]) is str:
                    self.watch_figs = [plt.figure(num=afig) for afig in watch_figs]

            save_dir = save_dir or P.tmp().string
            self.save_name = timestamp(name=save_name)
            self.save_dir = save_dir
            self.kwargs = kwargs
            self.counter = 0
            self.max = max_calls

        def add(self, fignames=None, names=None, **kwargs):
            print(f"Saver added frame number {self.counter}", end='\r')
            self.counter += 1
            plt.pause(self.delay * 0.001)
            if self.counter > self.max:
                print('Turning off IO')
                plt.ioff()

            if fignames:  # name sent explicitly
                self.watch_figs = [plt.figure(figname) for figname in fignames]
            else:  # tow choices:
                if self.watch_figs is None:  # None exist ==> add all
                    figure_names = plt.get_figlabels()  # add all.
                    self.watch_figs = [plt.figure(k) for k in figure_names]
                else:  # they exist already.
                    pass

            if names is None:  # individual save name, useful for PNG.
                names = [timestamp(name=a_figure.get_label()) for a_figure in self.watch_figs]

            for afig, aname in zip(self.watch_figs, names):
                self._save(afig, aname, **kwargs)

        def _save(self, *args, **kwargs):
            pass

    class Null(GenericSave):
        """ Use this when you do not want to save anything. This class will help plot to work faster
        by removing lines of previous plot, so you get live animation cheaply.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fname = self.save_dir

        def finish(self):
            print(f"Nothing saved by {self}")
            return self.fname

    class PDF(GenericSave):
        """For pdf, you just need any figure manager, [update, clear, accumalate], preferabbly fastest.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from matplotlib.backends.backend_pdf import PdfPages
            self.fname = os.path.join(self.save_dir, self.save_name + '.pdf')
            self.pp = PdfPages(self.fname)

        def _save(self, a_fig, a_name, bbox_inches='tight', pad_inches=0.3, **kwargs):
            self.pp.savefig(a_fig, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

        def finish(self, open_result=True):
            print(f"Saving results ...")
            self.pp.close()
            print(f"PDF Saved @", P(self.fname).absolute().as_uri())
            if open_result:
                import webbrowser as wb
                # chrome_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe".replace('\\', '/')
                # wb.register('chrome', None, wb.BackgroundBrowser(chrome_path))
                # wb.get('chrome').open(self.fname)
                wb.open(self.fname)
            return self

    class PNG(GenericSave):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_dir = os.path.join(self.save_dir, self.save_name)
            os.makedirs(self.save_dir, exist_ok=True)
            self.fname = self.save_dir

        def _save(self, afigure, aname, dpi=150, **kwargs):
            aname = P(aname).make_valid_filename()
            afigure.savefig(os.path.join(self.save_dir, aname), bbox_inches='tight', pad_inches=0.3,
                            dpi=dpi, **kwargs)

        def finish(self):
            print(f"PNGs Saved @", P(self.fname).absolute().as_uri())
            return self.fname

    class GIF(GenericSave):
        """Requirements: same axis must persist, only new objects are drawn inside it.
        This is not harsh as no one wants to add multiple axes on top of each other.
        Next, the objects drawn must not be removed, or updated, instead they should pile up in axis.

        # do not pass names in the add method. names will be extracted from figures.
        # usually it is smoother when adding animate=True to plot or imshow commands for GIF purpose

        Works for images only. Add more .imshow to the same axis, and that's it. imshow will conver up previous images.
        For lines, it will superimpose it and will look ugly.

        If you clear the axis, nothing will be saved. This should not happend.
        The class will automatically detect new lines by their "neo" labels.
        and add them then hide them for the next round.
        Limitation of ArtistAnimation: works on lines and images list attached to figure axes.
        Doesn't work on axes, unless you add large number of them. As such, titles are not incorporated etc.
        """

        def __init__(self, interval=100, **kwargs):
            super().__init__(**kwargs)
            from collections import defaultdict
            self.container = defaultdict(lambda: [])
            self.interval = interval
            self.fname = None  # determined at finish time.

        def _save(self, afigure, aname, cla=False, **kwargs):
            fig_list = self.container[afigure.get_label()]
            subcontainer = []

            search = FigureManager.findobj(afigure, 'neo')
            for item in search:
                item.set_label('processed')
                item.set_visible(False)
                subcontainer += [item]

            fig_list.append(subcontainer)
            # if you want the method coupled with cla being used in main, then it add_line is required for axes.

        def finish(self):
            print("Saving the GIF ....")
            import matplotlib.animation as animation
            from matplotlib.animation import MovieWriter  # PillowWriter
            for idx, a_fig in enumerate(self.watch_figs):
                ims = self.container[a_fig.get_label()]
                if ims:
                    ani = animation.ArtistAnimation(a_fig, ims,
                                                    interval=self.interval, blit=True, repeat_delay=1000)
                    self.fname = os.path.join(self.save_dir, f'{a_fig.get_label()}_{self.save_name}.gif')
                    ani.save(self.fname, writer=MovieWriter(fps=4))
                    # if you don't specify the writer, it goes to ffmpeg by default then try others if that is not
                    # available, resulting in behaviours that is not consistent across machines.
                    print(f"GIF Saved @", P(self.fname).absolute().as_uri())
                else:
                    print(f"Nothing to be saved by GIF writer.")
                    return self.fname

    class GIFFileBased(GenericSave):
        def __init__(self, fps=4, dpi=100, bitrate=1800, _type='GIFFileBased', **kwargs):
            super().__init__(**kwargs)
            from matplotlib.animation import ImageMagickWriter as Writer
            extension = '.gif'
            if _type == 'GIFPipeBased':
                from matplotlib.animation import ImageMagickFileWriter as Writer
            elif _type == 'MPEGFileBased':
                from matplotlib.animation import FFMpegFileWriter as Writer
                extension = '.mp4'
            elif _type == 'MPEGPipeBased':
                from matplotlib.animation import FFMpegWriter as Writer
                extension = '.mp4'
            self.writer = Writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = os.path.join(self.save_dir, self.save_name + extension)
            assert self.watch_figs, "No figure was sent during instantiation of saver, therefore the writer cannot" \
                                    "be setup. Did you mean to use an autosaver?"
            self.writer.setup(fig=self.watch_figs[0], outfile=self.fname, dpi=dpi)

        def _save(self, afig, aname, **kwargs):
            self.writer.grab_frame(**kwargs)

        def finish(self):
            print('Saving results ...')
            self.writer.finish()
            print(f"Saved @", P(self.fname).absolute().as_uri())
            return self.fname

    class GIFPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGFileBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    """
    Parses the data automatically.
    For all subclasses, you need to provide a plotter class with animate method implemetend.
    You also need to have .fig attribute.
    """

    class GenericAuto(GenericSave):
        save_type = 'auto'

        def __init__(self, plotter_class, data, names_list=None, **kwargs):
            super().__init__(**kwargs)
            self.plotter_class = plotter_class
            self.data = data
            self.names_list = names_list
            self.kwargs = kwargs
            self.data_gen = None
            self.saver = None
            self.plotter = None

        def animate(self):
            def gen_function():
                for i in zip(*self.data):
                    yield i

            self.data_gen = gen_function
            self.plotter = self.plotter_class(*[piece[0] for piece in self.data], **self.kwargs)
            plt.pause(0.5)  # give time for figures to show up before updating them
            from tqdm import tqdm
            for idx, datum in tqdm(enumerate(self.data_gen())):
                self.plotter.animate(datum)
                self.saver.add(names=[self.names_list[idx]])
            self.saver.finish()

    class GIFAuto(GenericAuto):
        def __init__(self, plotter_class, data, interval=500, extension='gif', fps=4, **kwargs):
            super().__init__(plotter_class, data, **kwargs)
            writer = None
            from matplotlib import animation
            if extension == 'gif':
                writer = animation.PillowWriter(fps=fps)
            elif extension == 'mp4':
                writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=2500)

            def gen_function():
                for i in zip(*self.data):
                    yield i

            self.gen = gen_function
            self.plotter = self.plotter_class(*[piece[0] for piece in self.data], **kwargs)
            plt.pause(self.delay * 0.001)  # give time for figures to show up before updating them
            # noinspection PyTypeChecker
            self.ani = animation.FuncAnimation(self.plotter.fig, self.plotter.animate, frames=self.gen,
                                               interval=interval, repeat_delay=1500, fargs=None,
                                               cache_frame_data=True, save_count=10000)
            fname = f"{os.path.join(self.save_dir, self.save_name)}.{extension}"
            self.fname = fname
            self.ani.save(filename=fname, writer=writer)
            print(f"Saved @", P(self.fname).absolute().as_uri())

    class PDFAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.PDF(**kwargs)
            self.animate()

    class PNGAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.PNG(**kwargs)
            self.save_dir = self.saver.save_dir
            self.animate()
            self.fname = self.saver.fname

    class NullAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.Null(**kwargs)
            self.fname = self.saver.fname
            self.animate()

    class GIFFileBasedAuto(GenericAuto):
        def __init__(self, plotter_class, data, fps=4, dpi=150, bitrate=2500,
                     _type='GIFFileBasedAuto', **kwargs):
            super().__init__(**kwargs)
            from matplotlib.animation import ImageMagickWriter as Writer
            extension = '.gif'
            if _type == 'GIFPipeBasedAuto':
                from matplotlib.animation import ImageMagickFileWriter as Writer
            elif _type == 'MPEGFileBasedAuto':
                from matplotlib.animation import FFMpegFileWriter as Writer
                extension = '.mp4'
            elif _type == 'MPEGPipeBasedAuto':
                from matplotlib.animation import FFMpegWriter as Writer
                extension = '.mp4'

            self.saver = Writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = os.path.join(self.save_dir, self.save_name + extension)

            def gen_function():
                for i in zip(*data):
                    yield i

            self.data = gen_function
            self.plotter = plotter_class(*[piece[0] for piece in data], **kwargs)
            plt.pause(0.5)  # give time for figures to show up before updating them

            from tqdm import tqdm
            with self.saver.saving(fig=self.plotter.fig, outfile=self.fname, dpi=dpi):
                for datum in tqdm(self.data()):
                    self.plotter.animate(datum)
                    self.saver.grab_frame()
                    plt.pause(self.delay * 0.001)
            print(f"Results saved successfully @ {self.fname}")

    class GIFPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGFileBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)


class VisibilityViewer(FigureManager):
    artist = ['internal', 'external'][1]
    parser = ['internal', 'external'][1]
    stream = ['clear', 'accumulate', 'update'][1]
    """
    **Viewer Building Philosophy**: 
    
    Viewer should act as Saver and Browser:

    * How is the data viewed:
    
        * Can either be an artist himself, as in ImShow.
        * external artist is required to view the data (especially non image data)

    * Data parsing:

        * internal for loop to go through all the dataset passed.
            # Allows manual control over parsing.
        * external for loop. It should have add method.
            # Manual control only takes place after the external loop is over. #TODO parallelize this.

    * Refresh mechanism.

        * Clear the axis. (slowest, but easy on memory)
        * accumulate, using visibility to hide previous axes. (Fastest but memory intensive)
        * The artist has an update method. (best)

    The artist has to have:
    
        * fig, ax, txt attributes. ax and txt should be lists.
        * the ax and txt attributes should always belong to the same figure.
    
    Here and in all Visibility classes, the artist is assumed to be always creating new axes along the way.
    """

    def __init__(self, artist=None, hide_artist_axes=True):
        """
        This class works on hiding axes shown on a plot, so that a new plot can be drawn.
        Hiding is done via the method `add`.
        Thus, an external loop is required to parse through the plots one by one.
        Once the entire loop is finished, you can browse through the plots with the keyboard
        Animation is done bia method `animate`

        :param artist: A class that draws on one figure. It should have `.fig` attribute.
                       Can either be passed during instantiation, or everytime when `add` is called.
        :param hide_artist_axes:
        """
        super().__init__()
        self.index = -1
        self.index_max = 0
        self.current = None
        self.axes_repo = []
        self.texts_repo = []
        self.fig = None
        if artist:
            self.fig = artist.fig
            self.fig.canvas.mpl_connect('key_press_event', self.process_key)
            self.add(artist=artist, hide_artist_axes=hide_artist_axes)

    def add(self, artist=None, increment_index=True, hide_artist_axes=True):
        if artist is not None:
            self.artist = artist
        if self.fig is None:
            self.fig = artist.fig
            self.fig.canvas.mpl_connect('key_press_event', self.process_key)

        if increment_index:
            self.index += 1
            self.index_max += 1
        self.current = self.index
        self.axes_repo.append(self.artist.ax if type(self.artist.ax) is list else self.artist.ax.tolist())
        self.texts_repo.append(self.artist.txt if type(self.artist.txt) is list else self.artist.txt.tolist())
        print(f"VViewer added plot number {self.index}", end='\r')
        if hide_artist_axes:
            self.hide_artist_axes()

    def hide_artist_axes(self):
        for ax in self.artist.ax:
            ax.set_visible(False)
        for text in self.artist.txt:
            text.set_visible(False)

    def finish(self):  # simply: undo the last hiding
        self.current = self.index
        self.animate()

    def animate(self):
        # remove current axes and set self.index as visible.
        for ax in self.axes_repo[self.current]:
            ax.set_visible(False)
        for text in self.texts_repo[self.current]:
            text.set_visible(False)
        for ax in self.axes_repo[self.index]:
            ax.set_visible(True)
        for text in self.texts_repo[self.index]:
            text.set_visible(True)
        self.current = self.index
        self.fig.canvas.draw()


class VisibilityViewerAuto(VisibilityViewer):
    def __init__(self, data=None, artist=None, memorize=False, transpose=True, save_type=SaveType.Null,
                 save_dir=None, save_name=None, delay=1,
                 titles=None, legends=None, x_labels=None, pause=True, **kwargs):
        """
        The difference between this class and `VisibilityViewer` is that here the parsing of data is done
        internally, hence the suffix `Auto`.

        :param data: shoud be of the form [[ip1 list], [ip2 list], ...]
                     i.e. NumArgsPerPlot x NumInputsForAnimation x Input (possible points x signals)
        :param artist: an instance of a class that subclasses `Artist`
        :param memorize: if set to True, then axes are hidden and shown again, otherwise, plots constructed freshly
                         every time they're shown (axes are cleaned instead of hidden)
        """
        self.kwargs = kwargs
        self.memorize = memorize
        self.max_index_memorized = 0

        if transpose:
            data = np.array(list(zip(*data)))
        self.data = data
        self.legends = legends
        if legends is None:
            self.legends = [f"Curve {i}" for i in range(len(self.data))]
        self.titles = titles if titles is not None else np.arange(len(self.data))
        self.lables = x_labels

        if artist is None:
            artist = Artist(*self.data[0], title=self.titles[0], legends=self.legends, create_new_axes=True,
                            **kwargs)
        else:
            artist.plot(*self.data[0], title=self.titles[0], legends=self.legends)
            if memorize:
                assert artist.create_new_axes is True, "Auto Viewer is based on hiding and showing and requires new " \
                                                       "axes from the artist with every plot"
        self.artist = artist
        super().__init__(artist=self.artist, hide_artist_axes=False)
        self.index_max = len(self.data)
        self.pause = pause
        self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name,
                               delay=delay, fps=1000 / delay)
        self.fname = None

    def animate(self):
        for i in range(self.index, self.index_max):
            datum = self.data[i]
            if self.memorize:  # ==> plot and use .add() method
                if self.index > self.max_index_memorized:  # a new plot never done before
                    self.hide_artist_axes()
                    self.artist.plot(*datum, title=self.titles[i], legends=self.legends)
                    self.add(increment_index=False, hide_artist_axes=False)  # index incremented via press_key manually
                    self.max_index_memorized += 1
                else:  # already seen this plot before ==> use animate method of parent class to hide and show,
                    # not plot and add
                    # print(f"current = {self.current}")
                    super().animate()
            else:
                self.fig.clf()  # instead of making previous axis invisible, delete it completely.
                self.artist.plot(*datum, title=self.titles[i], legends=self.legends)
                # replot the new data point on a new axis.
            self.saver.add()
            if self.pause:
                break
            else:
                self.index = i
        if self.index == self.index_max - 1 and not self.pause:  # arrived at last image and not in manual mode
            self.fname = self.saver.finish()

    @staticmethod
    def test():
        return VisibilityViewerAuto(data=np.random.randn(1, 10, 100, 3))


class ImShow(FigureManager):
    artist = ['internal', 'external'][0]
    parser = ['internal', 'external'][0]
    stream = ['clear', 'accumulate', 'update'][2]

    def __init__(self, *images_list: typing.Union[list, np.ndarray], sup_titles=None, sub_labels=None, labels=None,
                 save_type=SaveType.Null, save_name=None, save_dir=None, save_kwargs=None,
                 subplots_adjust=None, gridspec=None, tight=True, info_loc=None,
                 nrows=None, ncols=None, ax=None,
                 figsize=None, figname='im_show', figpolicy=FigurePolicy.add_new,
                 auto_brightness=True, delay=200, pause=False,
                 **kwargs):
        """
        :param images_list: arbitrary number of image lists separated by comma, say N.
        :param sup_titles: Titles for frames. Must have a length equal to number of images in each list, say M.
        :param sub_labels: Must have a length = M, and in each entry there should be N labels.
        :param labels: if labels are sent via this keyword, they will be repeated for all freames.
        :param save_type:
        :param save_dir:
        :param save_name:
        :param nrows:
        :param ncols:
        :param delay:
        :param kwargs: passed to imshow

        Design point: The expected inputs are made to be consistent with image_list passed. Labels and titles passed
        should have similar structure. The function internally process them.

        Tip: Use np.arrray_split to get sublists and have multiple plots per frame. Useful for very long lists.
        """
        super(ImShow, self).__init__(info_loc=info_loc)

        num_plots = len(images_list)  # Number of images in each plot
        self.num_plots = num_plots
        lengths = [len(images_list[i]) for i in range(num_plots)]
        self.index_max = min(lengths)
        nrows, ncols = self.get_nrows_ncols(num_plots, nrows, ncols)

        # Pad zero images for lists that have differnt length from the max.
        # images_list = list(images_list)  # now it is mutable, unlike tuple.
        # for i, a_list in enumerate(images_list):
        #     diff = self.num_images - len(a_list)
        #     if diff > 0:
        #         for _ in range(diff):
        #             if type(a_list) is list:
        #                 a_list.append(np.zeros_like(a_list[0]))
        #             else:  # numpy array
        #                 a_list = np.concatenate([a_list, [a_list[0]]])
        #     images_list[i] = a_list
        # # As far as labels are concerned, None is typed, if length is passed.

        # axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        # axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        # bnext = Button(axnext, 'Next')
        # bnext.on_clicked(callback.next)
        # bprev = Button(axprev, 'Previous')
        # bprev.on_clicked(callback.prev)

        if sup_titles is None:
            sup_titles = [str(i) for i in np.arange(self.index_max)]

        if labels:
            sub_labels = [[a_label for _ in np.arange(self.index_max)] for a_label in labels]
        elif sub_labels is None:
            sub_labels = [[str(i) for i in np.arange(self.index_max)] for _ in range(self.num_plots)]

        self.image_list = images_list
        self.sub_labels = sub_labels
        self.titles = sup_titles
        self.delay = delay
        self.fname = None
        self.kwargs = kwargs
        self.event = None
        self.cmaps = Cycle(plt.colormaps())
        self.cmaps.set('viridis')
        self.auto_brightness = auto_brightness
        if ax is None:
            self.figpolicy = figpolicy
            self.fig = self.get_fig(figname=figname,
                                    figsize=(14, 9) if figsize is None else figsize, facecolor='white')
            if figsize is None:
                plt.get_current_fig_manager().full_screen_toggle()
                # .window.showMaximized()  # state('zoom')
                # plt.get_current_fig_manager().window.setGeometry(800,70,1000,900)
            if gridspec is not None:
                gs = self.fig.add_gridspec(gridspec[0])
                self.ax = []
                for ags in gs[1:]:
                    self.ax.append(self.fig.add_subplot(gs[ags[0], ags[1]]))
            else:
                self.ax = self.fig.subplots(nrows=nrows, ncols=ncols)
        else:
            self.ax = ax
            try:
                self.fig = ax[0].figure
            except TypeError:  # Not subscriptable, single axis.
                self.fig = ax.figure

        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.fig.canvas.mpl_connect("pick_event", self.annotate)
        if tight:
            self.fig.tight_layout()
        if subplots_adjust is not None:
            self.fig.subplots_adjust(**subplots_adjust)

        # if save_type.parser == "internal":
        #     raise TypeError("Requires external data parser")
        if save_kwargs is None:
            save_kwargs = {}
        self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name,
                               delay=delay, fps=1000 / delay, **save_kwargs)
        if nrows == 1 and ncols == 1:
            self.ax = [self.ax]  # make a list out of it.
        else:
            self.ax = self.ax.ravel()  # make a 1D  list out of a 2D array.
        for an_ax in self.ax:
            # an_ax.set_xticks([])
            # an_ax.set_yticks([])
            self.toggle_ticks(an_ax, state=False)
        self.transposed_images = [images for images in zip(*images_list)]
        self.transposed_sublabels = [labels for labels in zip(*sub_labels)]

        self.ims = []  # container for images.
        self.pause = pause
        self.index = 0
        self.animate()

    def animate(self):
        for i in range(self.index, self.index_max):
            for j, (an_image, a_label, an_ax) in enumerate(zip(self.transposed_images[i],
                                                               self.transposed_sublabels[i],
                                                               self.ax)):
                # with zipping, the shortest of the three, will stop the loop.
                if i == 0 and self.ims.__len__() < self.num_plots:
                    im = an_ax.imshow(an_image, animated=True, **self.kwargs)
                    self.ims.append(im)
                else:
                    self.ims[j].set_data(an_image)
                if self.auto_brightness:
                    self.ims[j].norm.autoscale(an_image)
                an_ax.set_xlabel(f'{a_label}')
            self.fig.suptitle(self.titles[i], fontsize=8)
            self.saver.add(names=[self.titles[i]])
            if self.pause:
                break
            else:
                self.index = i
        if self.index == self.index_max - 1 and not self.pause:  # arrived at last image and not in manual mode
            self.fname = self.saver.finish()

    def annotate(self, event, axis=None, data=None):
        for ax in self.ax:
            super().annotate(event, axis=ax, data=ax.images[0].get_array())

    @classmethod
    def from_saved_images_path_lists(cls, *image_list, **kwargs):
        images = []
        sub_labels = []
        for alist in image_list:
            image_subcontainer = []
            label_subcontainer = []
            for an_image in alist:
                image_subcontainer.append(plt.imread(an_image))
                label_subcontainer.append(P(an_image).name)
            images.append(image_subcontainer)
            sub_labels.append(label_subcontainer)
        return cls(*images, sub_labels=sub_labels, **kwargs)

    @classmethod
    def from_directories(cls, *directories, extension='png', **kwargs):
        paths = []
        for a_dir in directories:
            paths.append(P(a_dir).search(f"*.{extension}", win_order=True))
        return cls.from_saved_images_path_lists(*paths, **kwargs)

    @classmethod
    def from_saved(cls, *things, **kwargs):
        example_item = things[0]
        if isinstance(example_item, list):
            return cls.from_saved_images_path_lists(*things, **kwargs)
        else:
            return cls.from_directories(*things, **kwargs)

    @staticmethod
    def cm(im, nrows=3, ncols=7, **kwargs):
        """ Useful for looking at one image in multiple cmaps

        :param im:
        :param nrows:
        :param ncols:
        :param kwargs:
        :return:
        """
        styles = plt.colormaps()
        colored = [plt.get_cmap(style)(im) for style in styles]
        splitted = np.array_split(colored, nrows * ncols)
        labels = np.array_split(styles, nrows * ncols)
        _ = ImShow(*splitted, nrows=nrows, ncols=ncols, sub_labels=labels, **kwargs)
        return colored

    @staticmethod
    def imagesc():
        # https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
        # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
        pass

    @staticmethod
    def test():
        return ImShow(*np.random.randn(12, 101, 100, 100))

    @classmethod
    def complex(cls, data, pause=True, **kwargs):
        return cls(data.real, data.imag, np.angle(data), abs(data), labels=['Real Part', 'Imaginary Part',
                                                                            'Angle in Radians', 'Absolute Value'],
                   pause=pause, **kwargs)

    @staticmethod
    def resize(path, m, n):
        # Experimental.assert_package_installed("skimage")
        from skimage.transform import resize
        image = plt.imread(path)
        image_resized = resize(image, (m, n), anti_aliasing=True)
        plt.imsave(path, image_resized)


class Artist(FigureManager):
    def __init__(self, *args, ax=None, figname='Graph', title='', label='curve', style='seaborn',
                 create_new_axes=False, figpolicy=FigurePolicy.add_new, figsize=(7, 4), **kwargs):
        super().__init__(figpolicy=figpolicy)
        self.style = style
        # self.kwargs = kwargs
        self.title = title
        self.args = args
        self.line = self.cursor = self.check_b = None
        if ax is None:  # create a figure
            with plt.style.context(style=self.style):
                self.fig = self.get_fig(figname, figsize=figsize)
        else:  # use the passed axis
            self.ax = ax
            self.fig = ax[0].figure

        if len(args):  # if there's something to plot in the init
            if not ax:  # no ax sent but we need to plot, we need an ax, plot will soon call get_axes.
                self.create_new_axes = True  # just for the first time in this init method.
            self.plot(*self.args, label=label, title=title, **kwargs)
        else:  # nothing to be plotted in the init
            if not create_new_axes:  # are we going to ever create new axes?
                self.create_new_axes = True  # if not then let's create one now.
                self.get_axes()
        self.create_new_axes = create_new_axes
        self.visibility_ax = [0.01, 0.05, 0.2, 0.15]
        self.txt = []

    def plot(self, *args, **kwargs):
        self.get_axes()
        self.accessorize(*args, **kwargs)

    def accessorize(self, *args, legends=None, title=None, **kwargs):
        self.line = self.ax[0].plot(*args, **kwargs)
        if legends is not None:
            self.ax[0].legend(legends)
        if title is not None:
            self.ax[0].set_title(title)
        self.ax[0].grid('on')

    def get_axes(self):
        if self.create_new_axes:
            axis = self.fig.subplots()
            self.ax = np.array([axis])
        else:  # use same old axes
            pass

    def suptitle(self, title):
        self.txt = [self.fig.text(0.5, 0.98, title, ha='center', va='center', size=9)]

    def visibility(self):
        from matplotlib.widgets import CheckButtons
        self.fig.subplots_adjust(left=0.3)
        self.visibility_ax[-1] = 0.05 * len(self.ax.lines)
        rax = self.fig.add_axes(self.visibility_ax)
        labels = [str(line.get_label()) for line in self.ax.lines]
        visibility = [line.get_visible() for line in self.ax.lines]
        self.check_b = CheckButtons(rax, labels, visibility)

        def func(label):
            index = labels.index(label)
            self.ax.lines[index].set_visible(not self.ax.lines[index].get_visible())
            self.fig.canvas.draw()

        self.check_b.on_clicked(func)

    @staticmethod
    def styler(plot_gen):
        styles = plt.style.available
        for astyle in styles:
            with plt.style.context(style=astyle):
                plot_gen()
                plt.title(astyle)
                plt.pause(1)
                plt.cla()
