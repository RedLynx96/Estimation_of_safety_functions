
# Import necessary libraries
import tkinter as tk
from tkinter import Tk, Label, Button, filedialog, messagebox, ttk
from PIL import Image, ImageTk

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
        #master.geometry("600x400")  # Adjusted window size for better layout
        master.resizable(True, True)  # Allow window to be resizable

        # Initialize variables
        self.mat_data = None
        self.models = {}  # Dictionary to store loaded models
        self.predictions = {}  # Dictionary to store predictions per model
        self.classical_prediction = None
        self.xVectors = None
        self.Ic = np.linspace(0,1,1000)
        self.y = None

        # Configure grid layout for the master window
        master.grid_rowconfigure(0, weight=0)  # Label row
        master.grid_rowconfigure(1, weight=0)  # Button frame row
        master.grid_rowconfigure(2, weight=1)  # Plot frame row
        master.grid_columnconfigure(0, weight=1)

        # UI Elements
        self.label = Label(master, text="Load a '.mat' file to predict the safety function.")
        self.label.grid(row=0, column=0, pady=10, padx=10)

        # Frame for buttons to organize layout
        self.button_frame = tk.Frame(master)
        self.button_frame.grid(row=1, column=0, pady=5, padx=10, sticky='ew')

        # Configure grid for button_frame with three columns
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
            self.button_frame,
            text="Open .mat File",
            command=self.open_mat_file,
            state=tk.NORMAL,  # Initially enabled
            width=20
        )
        self.load_mat_button.grid(row=0, column=0, padx=5, pady=5)

        # Load Pre-trained Models Button
        self.load_model_button = Button(
            self.button_frame,
            text="Load Pre-trained Models",
            command=self.load_models_panel,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.load_model_button.grid(row=0, column=1, padx=5, pady=5)

        # Run Prediction and Classical Computation Buttons
        self.predict_button = Button(
            self.button_frame,
            text="Run Prediction",
            command=self.run_prediction,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.predict_button.grid(row=1, column=0, padx=5, pady=5)

        self.classical_computation_button = Button(
            self.button_frame,
            text="Classical Computation",
            command=self.Classical_computation,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.classical_computation_button.grid(row=1, column=1, padx=5, pady=5)

        # Visualize Safety Function Button
        self.visualize_button = Button(
            self.button_frame,
            text="Visualize Safety Function",
            command=self.visualize_safety_function,
            state=tk.DISABLED,  # Initially disabled
            width=20
        )
        self.visualize_button.grid(row=2, column=0, padx=5, pady=5)

        # Save Plot as Image Button
        self.save_plot_button = Button(
            self.button_frame,
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

        # Initialize canvas variable
        self.canvas = None

        print("GUI initialized. Ready to load models and .mat file.")

    def open_mat_file(self):
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
                    self.y = self.mat_data['y'] # Optional: Load 'y' data if available for the classical computation
                    self.y = self.y.flatten()
                    if len(self.y) >= 1000:
                        self.y = self.y[:1000]
                    else:
                        messagebox.showwarning("Warning", "the map function y must be sampled over 1000 values.")
                except:
                    self.y = None
                
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

                self.check_ready()
                self.label.config(text="MAT file loaded with filtered length of " + str(self.xVectors.shape[0]) + " points. \n" +
                                  "Please load models for the prediction.")

                # Refresh the canvas to display the new plot
            except Exception as e:
                print(f"Failed to load MAT file: {e}")
                messagebox.showerror("Error", f"Failed to load MAT file: {e}")

    def load_models_panel(self):
        """
        Opens a new window with options to select predefined models or a custom model.
        """
        # Create a new top-level window for model selection
        selection_window = tk.Toplevel(self.master)
        selection_window.title("Select Models to Load")
        selection_window.geometry("300x400")

        # List of predefined models
        predefined_models = ["2000", "1000", "500", "250", "100", "50", "25"]
        self.selected_models_vars = {model: tk.BooleanVar() for model in predefined_models}
        self.custom_model_var = tk.BooleanVar()
        self.custom_model_path = tk.StringVar()

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
        for idx, model in enumerate(predefined_models):
            cb = ttk.Checkbutton(scrollable_frame, text=f"Model {model}", variable=self.selected_models_vars[model])
            cb.pack(anchor='w', padx=10, pady=2)

        # Checkbox for custom model
        custom_cb = ttk.Checkbutton(scrollable_frame, text="Custom Model", variable=self.custom_model_var, command=lambda: self.toggle_custom_selection(scrollable_frame))
        custom_cb.pack(anchor='w', padx=10, pady=10)

        # Entry and button for custom model selection (initially disabled)
        self.custom_entry = ttk.Entry(scrollable_frame, textvariable=self.custom_model_path, state='disabled', width=30)
        self.custom_entry.pack(anchor='w', padx=30, pady=2)

        browse_button = ttk.Button(scrollable_frame, text="Browse", command=self.browse_custom_model, state='disabled')
        browse_button.pack(anchor='w', padx=30, pady=2)
        self.browse_button = browse_button  # Store reference to enable/disable

        # Load button to confirm selection
        confirm_button = ttk.Button(scrollable_frame, text="Load Selected Models", command=lambda: self.confirm_selection(selection_window))
        confirm_button.pack(pady=20)

    def toggle_custom_selection(self, parent):
        if self.custom_model_var.get():
            self.custom_entry.config(state='normal')
            self.browse_button.config(state='normal')
        else:
            self.custom_entry.config(state='disabled')
            self.browse_button.config(state='disabled')
            self.custom_model_path.set('')

    def browse_custom_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Custom Pre-trained Model",
            filetypes=[("H5 files", "*.h5;*.hdf5"), ("All files", "*.*")]
        )
        if model_path:
            self.custom_model_path.set(model_path)

    def confirm_selection(self, window):
        selected = []
        # Add predefined selected models
        for model, var in self.selected_models_vars.items():
            if var.get():
                # Define the path for predefined models
                # Adjust the path as per your project's directory structure
                model_file = f"pretrained_models/{model}/{model}_chk.hdf5"
                selected.append(model_file)

        # Add custom model if selected
        if self.custom_model_var.get():
            custom_path = self.custom_model_path.get()
            if not custom_path:
                messagebox.showerror("Error", "Please select a custom model file.")
                return
            selected.append(custom_path)

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

    def run_prediction(self):
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
            self.predictions = {}

            for model_name, model in app.models.items():
                if self.xVectors.shape[0] >= int(model_name):
                    x_input = np.expand_dims(self.xVectors[:int(model_name), :], axis=0)  # Shape: (1, i, 2)
                    x_input = np.tile(x_input, (len(self.Ic), 1, 1))
                elif self.xVectors.shape[0] < int(model_name):
                    num_padding_rows = int(model_name) - self.xVectors.shape[0]
                    padding = np.full((num_padding_rows, 2), -1)
                    x_input = np.expand_dims(np.vstack((self.xVectors, padding)), axis=0)  # Shape: (1, i, 2)
                    x_input = np.tile(x_input, (len(self.Ic), 1, 1))
                
                self.predictions[model_name] = model.predict([x_input,self.Ic])

            self.label.config(text="Prediction completed.")
            print("Safety function prediction completed successfully.")
            self.check_ready()
        except Exception as e:
            print(f"Prediction failed: {e}")
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.label.config(text="")

    def Classical_computation(self):

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
                max_noise = np.max(noise) + 10e-15  # Add a small value to avoid zero noise

                return max_noise

            # Define other necessary variables and parameters
            NQ = len(self.Ic)
            Q = self.Ic

            # Calculate noise_max using the GetNoise function
            self.noise_max = GetNoise()  # Example value: 0.0335
            print(f"Maximum noise value: {self.noise_max}")
            iteraciones = 25  # safe set iterations

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

        except Exception as e:
            print(f"Classical computation failed: {e}")
            messagebox.showerror("Error", f"Classical computation failed: {e}")
            self.label.config(text="")

    def visualize_safety_function(self):
        try:
            # Create a new figure for plotting
            fig, ax = plt.subplots(figsize=(6, 4))

            # Plot Classical Computation if available
            if self.classical_prediction is not None:
                ax.plot(self.Ic, self.classical_prediction, label='Classical Computation', color='green')

            for model_name, model in self.models.items():
                ax.plot(self.Ic, self.predictions[model_name], label=model_name, alpha=0.7)

            # Configure plot aesthetics
            ax.set_title('Estimations from the Dataset', fontsize=14)
            ax.set_xlabel('$Q$', fontsize=12)
            ax.set_ylabel('$U_\infty$', fontsize=12)
            ax.legend(loc='best', fontsize='small')
            ax.grid(True)
            plt.tight_layout()

            # Clear any existing plots in the plot_frame
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Embed the new plot into the Tkinter GUI
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Close the standalone plot window to prevent it from appearing
            plt.close(fig)

            print("Safety function visualized successfully.")
            #self.label.config(text="Safety function visualized.")
        except Exception as e:
            print(f"Failed to visualize safety function: {e}")
            messagebox.showerror("Error", f"Failed to visualize safety function: {e}, please run prediction first.")

    def save_plot(self):
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
        Enable buttons based on loaded components.
        """
        # If mat_data is loaded but models are not loaded yet
        if self.mat_data is not None and not self.models:
            self.load_model_button.config(state=tk.NORMAL)
            self.classical_computation_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.DISABLED)
            self.visualize_button.config(state=tk.DISABLED)

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
            self.visualize_button.config(state=tk.NORMAL)
            self.save_plot_button.config(state=tk.NORMAL)
            self.label.config(text="U\u221E converged at iteration " + str(self.kend) + ".\n" +
                            "Estimated noise value: " + str(np.round(self.noise_max, 4)) + ".\n" +
                            "Ready to predict over the array of length "  + str(self.xVectors.shape[0]) + ".")

# Initialize and run the GUI
if __name__ == "__main__":
    root = Tk()
    app = SafetyFunctionGUI(root)
    root.mainloop()