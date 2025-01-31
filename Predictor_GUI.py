# Import necessary libraries
import tkinter as tk
from tkinter import Tk, Label, Button, filedialog, messagebox, ttk, colorchooser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tensorflow.keras.models import load_model

import scipy.io as sp
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore")

# Import Helper_FCN from the lib package
# Ensure that 'lib' is in your PYTHONPATH or current directory
try:
    from lib import Helper_FCN as HFCN
except ImportError as e:
    print(f"Error importing Helper_FCN: {e}")
    HFCN = None  # Handle this appropriately in your code

class SafetyFunctionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Safety Function Predictor")
        master.resizable(True, True)  # Allow window to be resizable

        # Initialize variables
        self.mat_data = None
        self.models = {}  # Dictionary to store loaded models
        self.predictions = {}  # Dictionary to store predictions per model
        self.classical_prediction = None
        self.xVectors = None
        self.Ic = np.linspace(0, 1, 1000)
        self.y = None
        self.kend = None
        self.noise_max = None

        # Initialize plot options
        self.plot_options = {
            "Classical Computation": {
                "visible": tk.BooleanVar(value=True),
                "color": "green",
                "linestyle": tk.StringVar(value='-'),  # Initialize linestyle
                "linewidth": tk.DoubleVar(value=1.5)   # Initialize linewidth
            },
            # Models will be added dynamically after visualization
        }

        # Flags to check if controls have been added
        self.model_controls_added = False
        self.classical_controls_added = False

        # Initialize color cycle using tab10
        self.color_cycle = plt.cm.tab10.colors
        self.next_color_index = 0  # To keep track of the next color to assign

        # Configure grid layout for the master window
        master.grid_rowconfigure(0, weight=0)  # Label row
        master.grid_rowconfigure(1, weight=0)  # Button frame row
        master.grid_rowconfigure(2, weight=1)  # Plot frame row
        master.grid_rowconfigure(3, weight=1)  # Options frame row
        master.grid_columnconfigure(0, weight=1)

        # UI Elements
        self.label = Label(master, text="Load a '.mat' file to predict the safety function.")
        self.label.grid(row=0, column=0, pady=10, padx=10)

        # Frame for buttons to organize layout
        self.button_frame = tk.Frame(master)
        self.button_frame.grid(row=1, column=0, pady=5, padx=10, sticky='ew')

        # Configure grid for button_frame with two columns
        self.button_frame.grid_columnconfigure(0, weight=1)  # Left spacer
        self.button_frame.grid_columnconfigure(1, weight=1)  # Right spacer

        # Internal container to hold buttons, centered
        buttons_container = tk.Frame(self.button_frame)
        buttons_container.grid(row=0, column=0, columnspan=2, pady=5)  # Spanning two columns

        # Configure buttons_container to have two columns
        buttons_container.grid_columnconfigure(0, weight=1)
        buttons_container.grid_columnconfigure(1, weight=1)

        # Open .mat File Button
        self.load_mat_button = Button(
            buttons_container,
            text="Open .mat File",
            command=self.open_mat_file,
            state=tk.NORMAL,  # Initially enabled
            width=20
        )
        self.load_mat_button.grid(row=0, column=0, padx=5, pady=5)

        # Load Pre-trained Models Button
        self.load_model_button = Button(
            buttons_container,
            text="Load Pre-trained Models",
            command=self.load_models_panel,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.load_model_button.grid(row=0, column=1, padx=5, pady=5)

        # Run Prediction and Classical Computation Buttons
        self.predict_button = Button(
            buttons_container,
            text="Run Prediction",
            command=self.run_prediction,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.predict_button.grid(row=1, column=0, padx=5, pady=5)

        self.classical_computation_button = Button(
            buttons_container,
            text="Classical Computation",
            command=self.Classical_computation,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.classical_computation_button.grid(row=1, column=1, padx=5, pady=5)

        # Visualize Safety Function Button
        self.visualize_button = Button(
            buttons_container,
            text="Visualize Safety Function",
            command=self.visualize_safety_function,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.visualize_button.grid(row=2, column=0, padx=5, pady=5)

        # Save Plot as Image Button
        self.save_plot_button = Button(
            buttons_container,
            text="Save Plot as Image",
            command=self.save_plot,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.save_plot_button.grid(row=2, column=1, padx=5, pady=5)

        # Frame to hold the plot
        self.plot_frame = tk.Frame(master)
        self.plot_frame.grid(row=2, column=0, pady=10, padx=10, sticky='nsew')

        # Configure grid for plot_frame to make it expandable
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        # Initialize figure and axes
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame for plot options
        self.options_frame = tk.Frame(master)
        self.options_frame.grid(row=3, column=0, pady=5, padx=10, sticky='ew')

        # Create control_container to hold control frames within a scrollable canvas
        self.control_canvas = tk.Canvas(self.options_frame)
        self.control_scrollbar = ttk.Scrollbar(self.options_frame, orient="vertical", command=self.control_canvas.yview)
        self.control_container = tk.Frame(self.control_canvas)

        self.control_container.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )

        self.control_canvas.create_window((0, 0), window=self.control_container, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)

        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_scrollbar.pack(side="right", fill="y")

        # Create a frame for the save_predictions_button
        self.save_button_container = tk.Frame(self.options_frame)
        self.save_button_container.pack(anchor='w', pady=5, padx=5)

        # Initialize save_predictions_button in save_button_container, but not packed yet
        self.save_predictions_button = Button(
            self.save_button_container,
            text="Save Predictions as .mat",
            command=self.save_predictions_as_mat,
            state=tk.DISABLED,  # Initially disabled
            width=25
        )
        # Initially not packed

        # List to keep track of all control frames for dynamic arrangement
        self.control_frames = []

        # Bind the <Configure> event to rearrange controls on window resize
        self.master.bind("<Configure>", self.arrange_controls)

        print("GUI initialized. Ready to load models and .mat file.")

    def add_model_plot_options(self):
        """
        Adds visibility toggles, color change buttons, linestyle, and linewidth controls for each loaded model.
        Arranges them in a scrollable frame.
        """
        for model_name in self.models.keys():
            if model_name not in self.plot_options:
                # Assign unique color from tab10
                assigned_color = self.color_cycle[self.next_color_index % len(self.color_cycle)]
                self.next_color_index += 1  # Update for next model

                self.plot_options[model_name] = {
                    "visible": tk.BooleanVar(value=True),
                    "color": assigned_color,
                    "linestyle": tk.StringVar(value='-'),  # Initialize linestyle
                    "linewidth": tk.DoubleVar(value=1.5)   # Initialize linewidth
                }

                # Frame for each model's controls
                model_frame = tk.Frame(self.control_container, borderwidth=1, relief='groove', padx=5, pady=5)
                self.control_frames.append(model_frame)  # Add to the list

                # Model label
                model_label = Label(model_frame, text=model_name, font=('Arial', 10, 'bold'))
                model_label.grid(row=0, column=0, sticky='w')

                # Visibility Checkbox
                cb = tk.Checkbutton(
                    model_frame,
                    text="Show",
                    variable=self.plot_options[model_name]["visible"],
                    command=self.update_plot
                )
                cb.grid(row=1, column=0, sticky='w')

                # Buttons Frame
                buttons_subframe = tk.Frame(model_frame)
                buttons_subframe.grid(row=2, column=0, sticky='w', pady=2)

                # Change Color Button
                color_btn = Button(
                    buttons_subframe,
                    text="Color",
                    command=lambda mn=model_name: self.change_line_color(mn),
                    width=8
                )
                color_btn.grid(row=0, column=0, padx=2)

                # Change Linestyle Button
                linestyle_btn = Button(
                    buttons_subframe,
                    text="Linestyle",
                    command=lambda mn=model_name: self.change_line_style(mn),
                    width=8
                )
                linestyle_btn.grid(row=0, column=1, padx=2)

                # Change Linewidth Button
                linewidth_btn = Button(
                    buttons_subframe,
                    text="Linewidth",
                    command=lambda mn=model_name: self.change_line_width(mn),
                    width=8
                )
                linewidth_btn.grid(row=0, column=2, padx=2)

        self.model_controls_added = True  # Update the flag

    def add_classical_computation_controls(self):
        """
        Adds visibility toggles, color change buttons, linestyle, and linewidth controls for classical computation.
        Arranges them in a scrollable frame.
        """
        if self.classical_controls_added:
            return  # Prevent adding controls multiple times

        # Frame for Classical Computation controls
        classical_frame = tk.Frame(self.control_container, borderwidth=1, relief='groove', padx=5, pady=5)
        self.control_frames.append(classical_frame)  # Add to the list

        # Classical Computation label
        classical_label = Label(classical_frame, text="Classical Computation", font=('Arial', 10, 'bold'))
        classical_label.grid(row=0, column=0, sticky='w')

        # Visibility Checkbox
        cb = tk.Checkbutton(
            classical_frame,
            text="Show",
            variable=self.plot_options["Classical Computation"]["visible"],
            command=self.update_plot
        )
        cb.grid(row=1, column=0, sticky='w')

        # Buttons Frame
        buttons_subframe = tk.Frame(classical_frame)
        buttons_subframe.grid(row=2, column=0, sticky='w', pady=2)

        # Change Color Button
        color_btn = Button(
            buttons_subframe,
            text="Color",
            command=lambda: self.change_line_color("Classical Computation"),
            width=8
        )
        color_btn.grid(row=0, column=0, padx=2)

        # Change Linestyle Button
        linestyle_btn = Button(
            buttons_subframe,
            text="Linestyle",
            command=lambda: self.change_line_style("Classical Computation"),
            width=8
        )
        linestyle_btn.grid(row=0, column=1, padx=2)

        # Change Linewidth Button
        linewidth_btn = Button(
            buttons_subframe,
            text="Linewidth",
            command=lambda: self.change_line_width("Classical Computation"),
            width=8
        )
        linewidth_btn.grid(row=0, column=2, padx=2)

        self.classical_controls_added = True  # Update the flag

    def change_line_color(self, line_name):
        """
        Opens a color chooser dialog to select a new color for the specified line.
        Updates the plot with the new color.
        """
        # Open color chooser dialog
        color_code = colorchooser.askcolor(title=f"Choose color for {line_name}")
        if color_code and color_code[1]:
            # Update the color in plot_options
            self.plot_options[line_name]["color"] = color_code[1]
            # Update the plot with the new color
            self.update_plot()

    def change_line_style(self, line_name):
        """
        Opens a dialog to select a new linestyle for the specified line.
        Updates the plot with the new linestyle.
        """
        # Define available linestyles
        linestyles = ['-', '--', '-.', ':']
        # Create a new top-level window for linestyle selection
        linestyle_window = tk.Toplevel(self.master)
        linestyle_window.title(f"Select Linestyle for {line_name}")

        # Instruction label
        instruction = Label(linestyle_window, text="Choose a linestyle:")
        instruction.grid(row=0, column=0, columnspan=4, pady=10, padx=10)

        # Radio buttons for each linestyle
        for idx, ls in enumerate(linestyles):
            rb = tk.Radiobutton(
                linestyle_window,
                text=ls,
                variable=self.plot_options[line_name]["linestyle"],
                value=ls,
                command=self.update_plot
            )
            rb.grid(row=1, column=idx, sticky='w', padx=20)

    def change_line_width(self, line_name):
        """
        Opens a dialog to select a new linewidth for the specified line.
        Updates the plot with the new linewidth.
        """
        # Create a new top-level window for linewidth selection
        linewidth_window = tk.Toplevel(self.master)
        linewidth_window.title(f"Select Linewidth for {line_name}")

        # Instruction label
        instruction = Label(linewidth_window, text="Enter a linewidth value (e.g., 1.0):")
        instruction.grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        # Entry widget for linewidth
        linewidth_entry = tk.Entry(linewidth_window)
        linewidth_entry.grid(row=1, column=0, pady=5, padx=10)

        # Submit button
        def submit_linewidth():
            try:
                value = float(linewidth_entry.get())
                if value <= 0:
                    raise ValueError
                self.plot_options[line_name]["linewidth"].set(value)
                self.update_plot()
                linewidth_window.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a positive numerical value for linewidth.")

        submit_btn = Button(linewidth_window, text="Submit", command=submit_linewidth)
        submit_btn.grid(row=1, column=1, pady=5, padx=10)

    def update_plot(self):
        """
        Redraws the plot based on the current state of plot_options.
        """
        # Clear existing axes
        self.ax.clear()
        
        # Plot Classical Computation if visible
        if self.plot_options["Classical Computation"]["visible"].get() and self.classical_prediction is not None:
            try:
                self.ax.plot(
                    self.Ic,
                    self.classical_prediction,
                    label='Classical Computation',
                    color=self.plot_options["Classical Computation"]["color"],
                    linestyle=self.plot_options["Classical Computation"]["linestyle"].get(),
                    linewidth=self.plot_options["Classical Computation"]["linewidth"].get()
                )
            except AttributeError as e:
                messagebox.showerror("Plot Error", f"Error plotting Classical Computation: {e}")
                print(f"Plot Error: {e}")

        # Plot each model if visible
        for model_name, prediction in self.predictions.items():
            if model_name in self.plot_options and self.plot_options[model_name]["visible"].get():
                try:
                    self.ax.plot(
                        self.Ic,
                        prediction,
                        label=model_name,
                        color=self.plot_options[model_name]["color"],
                        linestyle=self.plot_options[model_name]["linestyle"].get(),
                        linewidth=self.plot_options[model_name]["linewidth"].get()
                    )
                except AttributeError as e:
                    messagebox.showerror("Plot Error", f"Error plotting {model_name}: {e}")
                    print(f"Plot Error for {model_name}: {e}")

        # Configure plot aesthetics
        self.ax.set_title('Estimations from the Dataset', fontsize=14)
        self.ax.set_xlabel('$Q$', fontsize=12)
        self.ax.set_ylabel('$U_\\infty$', fontsize=12)
        self.ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        
        # Redraw the canvas
        self.canvas.draw()

    def open_mat_file(self):
        """
        Opens a file dialog to select and load a .mat file.
        Processes the data and updates the GUI state.
        """
        if self.canvas:
            # Destroy the existing FigureCanvasTkAgg widget
            self.canvas.get_tk_widget().destroy()
            self.canvas = None  # Reset the canvas reference
            
        # Initialize variables
        self.mat_data = None
        self.models = {}  # Dictionary to store loaded models
        self.predictions = {}  # Dictionary to store predictions per model
        self.classical_prediction = None
        self.xVectors = None
        self.y = None
        self.kend = None
        self.noise_max = None

        # Reset plot options and plot
        self.plot_options = {
            "Classical Computation": {
                "visible": tk.BooleanVar(value=True),
                "color": "green",
                "linestyle": tk.StringVar(value='-'),  # Initialize linestyle
                "linewidth": tk.DoubleVar(value=1.5)   # Initialize linewidth
            },
            # Models will be added dynamically after visualization
        }
        self.classical_controls_added = False  # Reset the flag
        self.model_controls_added = False  # Reset the flag

        # Re-create the plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Disable and hide the save_predictions_button
        self.save_predictions_button.config(state=tk.DISABLED)
        if self.save_predictions_button.winfo_ismapped():
            self.save_predictions_button.pack_forget()

        # Clear any existing control frames
        for frame in self.control_frames:
            frame.destroy()
        self.control_frames.clear()

        messagebox.showinfo(
            "Reminder", 
            "Be sure that the variable in the .mat file is named: 'xVectors', and is shaped (1000,50,2)."
        )
        mat_path = filedialog.askopenfilename(
            title="Select .mat File",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
        )
        if mat_path:
            try:
                # Load and process the MAT file
                self.mat_data = sp.loadmat(mat_path)
                self.xVectors = self.mat_data['xVectors']

                try:
                    self.y = self.mat_data['y']  # Optional: Load 'y' data if available for the classical computation
                    self.y = self.y.flatten()
                    if len(self.y) >= 1000:
                        self.y = self.y[:1000]
                    else:
                        messagebox.showwarning("Warning", "The map function y must be sampled over 1000 values.")
                except KeyError:
                    self.y = None  # 'y' data is optional

                if self.xVectors.shape == (1000, 50, 2):
                    self.xVectors = np.nan_to_num(self.xVectors, nan=-1)  # Replace NaNs with -1
                    self.xVectors = np.reshape(self.xVectors, (1000 * 50, 2))
                    self.xVectors = self.xVectors[~np.any(self.xVectors == -1, axis=1)]
                
                    Separated_xVectors = []
                    for i in range(len(self.xVectors) - 1):
                        Separated_xVectors.append(self.xVectors[i])
                        if self.xVectors[i][1] != self.xVectors[i + 1][0]:
                            Separated_xVectors.append([-1, -1])
                    # Append the last row
                    Separated_xVectors.append(self.xVectors[-1])
                    # Convert result back to numpy array
                    self.xVectors = np.array(Separated_xVectors)
                else:
                    print('Array shaped' + str(self.xVectors.shape))

                # Reset predictions and classical computations
                self.predictions = {}
                self.classical_prediction = None
                self.kend = None
                self.noise_max = None

                # Clear any existing control frames
                for frame in self.control_frames:
                    frame.destroy()
                self.control_frames.clear()

                # Update the GUI to reflect the loaded data
                self.check_ready()
                self.label.config(text="MAT file loaded with filtered length of " + str(self.xVectors.shape[0]) + " points.\n" +
                                  "Please load models for the prediction.")

            except Exception as e:
                print(f"Failed to load MAT file: {e}")
                messagebox.showerror("Error", f"Failed to load MAT file: {e}")

    def load_models_panel(self):
        """
        Opens a new window with options to select predefined models.
        Removed all functionalities related to loading a custom model.
        """
        # Create a new top-level window for model selection
        selection_window = tk.Toplevel(self.master)
        selection_window.title("Select Models to Load")

        # List of predefined models
        predefined_models = ["2000", "1000", "500", "250", "100", "50", "25"]
        self.selected_models_vars = {model: tk.BooleanVar() for model in predefined_models}

        # Scrollable Frame in case of many models
        canvas = tk.Canvas(selection_window)
        scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create checkboxes for predefined models
        for model in predefined_models:
            cb = ttk.Checkbutton(scrollable_frame, text=f"Model {model}", variable=self.selected_models_vars[model])
            cb.pack(anchor='w', padx=10, pady=5)

        # Load button to confirm selection
        confirm_button = ttk.Button(scrollable_frame, text="Load Selected Models", command=lambda: self.confirm_selection(selection_window))
        confirm_button.pack(pady=20)

    def confirm_selection(self, window):
        """
        Loads the selected models and updates the GUI state.
        Only handles predefined models since custom model loading has been removed.
        """
        selected = []
        # Add predefined selected models
        for model, var in self.selected_models_vars.items():
            if var.get():
                # Define the path for predefined models
                # Adjust the path as per your project's directory structure
                model_file = f"pretrained_models/{model}/{model}_chk.hdf5"
                selected.append(model_file)

        if not selected:
            messagebox.showwarning("No Selection", "No models selected to load.")
            return

        # Attempt to load all selected models
        failed_models = []
        for model_path in selected:
            try:
                if HFCN is None:
                    raise ImportError("File Helper_FCN.py is not found in folder 'lib'.")

                loaded_model = load_model(model_path, custom_objects={'AsymmetricMSELoss': HFCN.AsymmetricMSELoss})
                model_name = model_path.split('/')[-1]
                model_name = model_name.split('_')[0]
                self.models[model_name] = loaded_model
                print(f"Loaded model: {model_name}")  # For debugging purposes

            except Exception as e:
                failed_models.append(model_path)
                print(f"Failed to load model '{model_path}': {e}")

        if failed_models:
            messagebox.showerror("Error", f"Failed to load the following models:\n" + "\n".join(failed_models))
        else:
            messagebox.showinfo("Success", f"Successfully loaded {len(selected)} model(s).")

        window.destroy()
        self.check_ready()

        # **No call to add_model_plot_options() here**
        # Controls will be added during visualization

    def run_prediction(self):
        """
        Runs predictions using the loaded models and updates the GUI state.
        """
        if self.mat_data is None or not self.models:
            print("Error: Please load both the models and the .mat file before prediction.")
            messagebox.showerror("Error", "Please load both the models and the .mat file before prediction.")
            return

        self.label.config(text="Running prediction...")
        self.master.update_idletasks()
        print("Running prediction...")

        self.predictions = {}  # Reset previous predictions

        try:
            if HFCN is None:
                raise ImportError("Helper_FCN is not available.")

            # Example processing based on the original code
            # Adjust according to your actual prediction requirements

            # Padding or trimming xVectors to match model input requirements
            for model_name, model in self.models.items():
                required_length = int(model_name)
                if self.xVectors.shape[0] >= required_length:
                    x_input = np.expand_dims(self.xVectors[:required_length, :], axis=0)  # Shape: (1, i, 2)
                    x_input = np.tile(x_input, (len(self.Ic), 1, 1))
                elif self.xVectors.shape[0] < required_length:
                    num_padding_rows = required_length - self.xVectors.shape[0]
                    padding = np.full((num_padding_rows, 2), -1)
                    x_input = np.expand_dims(np.vstack((self.xVectors, padding)), axis=0)  # Shape: (1, i, 2)
                    x_input = np.tile(x_input, (len(self.Ic), 1, 1))
                
                self.predictions[model_name] = model.predict([x_input, self.Ic])

            self.label.config(text="Prediction completed.")
            print("Safety function prediction completed successfully.")
            self.check_ready()

            # **No call to add_model_plot_options() here**
            # Controls will be added during visualization

        except Exception as e:
            print(f"Prediction failed: {e}")
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.label.config(text="")

    def Classical_computation(self):
        """
        Performs classical computation and updates the GUI state.
        """
        if self.xVectors is None or self.y is None:
            print("Error: 'xVectors' or 'y' data is missing.")
            messagebox.showerror("Error", "'xVectors' data or function 'y' is missing. Being 'y' the known function of the dynamical system shaped (1,1000).")
            return

        self.label.config(text="Running classical computation...")
        self.master.update_idletasks()
        print("Running classical computation...")

        try:
            # Define the GetNoise function
            def GetNoise():
                """
                y_func is the known function of the dynamical system f(x), therefore if xVectors are noised sampled points from y_func, the noise can be calculated.
                
                Parameters:
                - xVectors (np.ndarray): A NumPy array of shape [batch, points, 2],
                - y_func (np.ndarray): A 1D NumPy array representing y-values over a specified range of x-values.
                
                Returns:
                - float: The maximum vertical distance found.
                """

                # Step 1: Create an interpolation function for self.y
                interp_func = interp1d(self.Ic, self.y, kind='linear', bounds_error=False, fill_value='extrapolate')

                # Step 2: Filter out invalid samples where both x and y are -1
                # Create a boolean mask where rows are not [-1, -1]
                valid_mask = ~((self.xVectors[:, 0] == -1) & (self.xVectors[:, 1] == -1))

                # Apply the mask to obtain valid x and noisy y values
                valid_x = self.xVectors[valid_mask, 0]
                valid_noisy_y = self.xVectors[valid_mask, 1]

                # Step 4: Interpolate y-values at the valid x positions
                interpolated_y = interp_func(valid_x)

                # Step 5: Calculate the absolute noise for each valid sample
                noise = np.abs(valid_noisy_y - interpolated_y)

                # Step 5: Find the maximum noise value
                max_noise = np.max(noise) + 1e-14  # Add a small value to avoid zero noise

                return max_noise

            # Define other necessary variables and parameters
            NQ = len(self.Ic)
            Q = self.Ic

            # Calculate noise_max using the GetNoise function
            self.noise_max = GetNoise()  # Example value: 0.0335
            print(f"Maximum noise value: {self.noise_max}")
            iteraciones = 50  # safe set iterations

            # Interpolate y_func (self.Us) to have 1000 points
            interp_func = interp1d(
                self.Ic,
                self.y,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            Imagen = interp_func(Q)

            # Ensure Imagen is 1D
            Imagen = Imagen.flatten()

            # Define escala
            escala = 1 / (NQ -1)

            # Add noise
            ruido = np.arange(-self.noise_max, self.noise_max + escala, escala)
            Nruido = len(ruido)
            anchoruido = Nruido // 2

            Imagen_ruido = np.zeros((NQ, Nruido))

            for i in range(NQ):
                for s in range(Nruido):
                    Imagen_ruido[i, s] = self.y[i] + ruido[s]

            ymin = np.min(Imagen_ruido) - escala
            ymax = np.max(Imagen_ruido) + escala

            Imagen_ruidoy = np.arange(ymin, ymax + escala, escala)
            Nimagen_ruidoy = len(Imagen_ruidoy)
            controly = np.zeros(Nimagen_ruidoy)

            Us = np.zeros((iteraciones+1, NQ))

            IndImagen = np.round((Imagen - ymin) / escala).astype(int)

            self.kend = iteraciones

            for k in range(iteraciones):
                for m in range(Nimagen_ruidoy):
                    controly_posibles_inQ = np.abs(Imagen_ruidoy[m] - Q)
                    controly[m] = np.min(np.maximum(Us[k, :], controly_posibles_inQ))

                for i in range(NQ):
                    # Handle index boundaries to avoid out-of-bounds
                    lower = IndImagen[i] - anchoruido
                    upper = IndImagen[i] + anchoruido

                    # Clip indices to valid range
                    lower = max(lower, 0)
                    upper = min(upper, Nimagen_ruidoy)

                    Us[k + 1, i] = np.max(controly[lower:upper])

                if np.array_equal(Us[k + 1, :], Us[k, :]):
                    self.kend = k
                    print('Converge en la iteración: ', self.kend)
                    break

            Us = Us[:self.kend+1, :]

            # Store the classical computation result as the last iteration's Us
            self.classical_prediction = Us[self.kend, :]  # Shape: (NQ,)

            messagebox.showinfo("Success", "Classical computation completed successfully.")   
            self.label.config(text="Classical computation completed.")
            self.check_ready()

            # **Do not add classical computation controls here**
            # They will be added when visualizing
        except Exception as e:
            print(f"Classical computation failed: {e}")
            messagebox.showerror("Error", f"Classical computation failed: {e}")
            self.label.config(text="")

    def visualize_safety_function(self):
        """
        Visualizes the safety function based on the current predictions and classical computation.
        Adds model controls and classical computation controls in a scrollable frame after visualization.
        Plots all curves in a single click.
        """
        try:
            # Check if there's data to plot
            if self.classical_prediction is None and not self.predictions:
                messagebox.showerror("Error", "No data to visualize. Please run prediction or classical computation first.")
                return

            # Add model controls only once
            if not self.model_controls_added and self.models:
                self.add_model_plot_options()

            # Add classical computation controls only if computation was performed and controls not yet added
            if self.classical_prediction is not None and not self.classical_controls_added:
                self.add_classical_computation_controls()

            # Update the plot based on current plot options
            self.update_plot()

            # Show the save_predictions_button if there are predictions or classical_prediction
            if self.predictions or self.classical_prediction is not None:
                if not self.save_predictions_button.winfo_ismapped():
                    self.save_predictions_button.pack(anchor='w', padx=20, pady=2)
                self.save_predictions_button.config(state=tk.NORMAL)
            else:
                self.save_predictions_button.config(state=tk.DISABLED)

            # After adding controls, arrange them
            self.arrange_controls()

            print("Safety function visualized successfully.")
        except Exception as e:
            print(f"Failed to visualize safety function: {e}")
            messagebox.showerror("Error", f"Failed to visualize safety function: {e}, please run prediction first.")

    def arrange_controls(self, event=None):
        """
        Arranges the control frames dynamically based on the window's width.
        This method is called whenever the window is resized or manually after adding controls.
        """
        try:
            # Determine the available width for controls
            available_width = self.control_canvas.winfo_width()
            if available_width == 1:
                # Initial width can be 1, so skip arranging
                return

            # Define approximate width of each control frame (including padding)
            frame_width = 250  # Adjust as needed based on your design
            padding = 10  # Padding between frames

            # Calculate the number of columns that can fit
            num_columns = max(1, available_width // (frame_width + padding))

            # Clear the current grid
            for frame in self.control_frames:
                frame.grid_forget()

            # Re-arrange the frames in the grid
            for idx, frame in enumerate(self.control_frames):
                row = idx // num_columns
                col = idx % num_columns
                frame.grid(row=row, column=col, padx=5, pady=5, sticky='w')

            # Update the layout
            self.options_frame.update_idletasks()

        except Exception as e:
            print(f"Error arranging controls: {e}")
            # Optionally, show an error message or pass
            pass

    def save_predictions_as_mat(self):
        """
        Saves the predictions and classical computation results to a .mat file.
        """
        # Prompt user to select save location and filename
        save_path = filedialog.asksaveasfilename(
            defaultextension=".mat",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
            title="Save Predictions as .mat File"
        )
        if save_path:
            try:
                # Prepare data to save
                data_to_save = {}
                if self.predictions:
                    # Convert predictions to list or suitable format
                    for model_name, prediction in self.predictions.items():
                        data_to_save[f"prediction_{model_name}"] = prediction
                if self.classical_prediction is not None:
                    data_to_save["classical_prediction"] = self.classical_prediction
                # Optionally, include other data like Ic
                data_to_save["Ic"] = self.Ic
                # Save to .mat file
                sp.savemat(save_path, data_to_save)
                print(f"Predictions saved to {save_path}")
                messagebox.showinfo("Success", f"Predictions saved to {save_path}")
            except Exception as e:
                print(f"Failed to save predictions: {e}")
                messagebox.showerror("Error", f"Failed to save predictions: {e}")

    def save_plot(self):
        """
        Saves the current plot as an image file.
        """
        if self.canvas is None:
            print("Error: No plot available to save.")
            messagebox.showerror("Error", "No plot available to save.")
            return

        # Ensure that self.canvas has a figure attribute
        if not hasattr(self.canvas, 'figure'):
            print("Error: The canvas does not contain a figure to save.")
            messagebox.showerror("Error", "The canvas does not contain a figure to save.")
            return

        # Prompt user to select save location and filename
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Plot as Image"
        )
        if save_path:
            try:
                # Save the current figure from the canvas
                fig = self.canvas.figure
                fig.savefig(save_path)
                print(f"Plot saved to {save_path}")
                messagebox.showinfo("Success", f"Plot saved to {save_path}")
            except Exception as e:
                print(f"Failed to save plot: {e}")
                messagebox.showerror("Error", f"Failed to save plot: {e}")

    def check_ready(self):
        """
        Enables or disables buttons based on loaded components.
        """
        # If mat_data is loaded but models are not loaded yet
        if self.mat_data is not None and not self.models:
            self.load_model_button.config(state=tk.NORMAL)
            self.classical_computation_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.DISABLED)
            self.visualize_button.config(state=tk.DISABLED)
            self.save_plot_button.config(state=tk.DISABLED)

            print("MAT file loaded. Awaiting models.")

        # If models and mat_data are loaded
        if self.mat_data is not None and self.models:
            self.predict_button.config(state=tk.NORMAL)
            self.visualize_button.config(state=tk.NORMAL)
            self.save_plot_button.config(state=tk.NORMAL)
            self.classical_computation_button.config(state=tk.NORMAL)
            self.label.config(text="Models and MAT file loaded. Ready to run prediction or classical computation.")
            print("Models and MAT file loaded. Ready to run prediction or classical computation.")

        # If classical computation is done
        if self.classical_prediction is not None:
            # Enable visualization even if classical computation was done before models
            self.visualize_button.config(state=tk.NORMAL)
            self.save_plot_button.config(state=tk.NORMAL)
            self.label.config(text="Classical computation completed.\nReady to visualize safety functions.")

# Initialize and run the GUI
if __name__ == "__main__":
    root = Tk()
    app = SafetyFunctionGUI(root)
    root.mainloop()
