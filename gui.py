"""
gui.py - Modern Tkinter user interface for the Aviator Score Analyzer.

Uses CustomTkinter (CTk) for a modern, responsive look with dark/light
mode capabilities. Provides two input modes:
  - Manual: paste score text directly
  - Image / Screenshot: paste from clipboard or browse an image file,
    then extract scores via OCR (pytesseract / Tesseract)
"""

import io
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Optional, List

import customtkinter as ctk
from PIL import Image, ImageTk

from database import (
    setup_db,
    insert_scores,
    fetch_all_scores,
    fetch_recent_scores,
    clear_database,
    get_total_count,
)
from ml_model import train_and_predict_from_history, MIN_ROUNDS_REQUIRED
from ocr_utils import extract_scores_from_image


# --- Appearance Configuration ---
ctk.set_appearance_mode("System")  # "System", "Dark", or "Light"
ctk.set_default_color_theme("blue")


# --- Constants ---
WINDOW_TITLE = "Aviator Score Analyzer & Predictor"
WINDOW_WIDTH = 780
WINDOW_HEIGHT = 860
PAD = {"padx": 20, "pady": 12}
PAD_SMALL = {"padx": 14, "pady": 8}
THUMBNAIL_SIZE = (360, 220)
CORNER_RADIUS = 12
CORNER_RADIUS_SMALL = 8
BUTTON_HEIGHT = 44
BUTTON_HEIGHT_SMALL = 38

# Color palette
COLOR_PROCESS = "#2E7D32"
COLOR_PROCESS_HOVER = "#1B5E20"
COLOR_EXTRACT = "#E65100"
COLOR_EXTRACT_HOVER = "#BF360C"
COLOR_DANGER = "#D32F2F"
COLOR_DANGER_HOVER = "#B71C1C"
COLOR_INFO = "#1565C0"
COLOR_INFO_HOVER = "#0D47A1"
COLOR_PREDICTION = "#FF6F00"
COLOR_SUCCESS = "#2E7D32"
COLOR_ACCENT = "#1f6feb"

# Appearance-mode-aware colors (set later)
_listbox_bg: str = "#2b2b2b"
_listbox_fg: str = "#e0e0e0"


# ------------------------------------------------------------------
# Text-parsing helper (shared by both modes)
# ------------------------------------------------------------------

def parse_score_text(raw_text: str) -> List[float]:
    """
    Parse user-provided text into a list of float scores.

    Accepts scores separated by commas, spaces, newlines, or mixed delimiters.

    Parameters
    ----------
    raw_text : str
        Raw input string from the user.

    Returns
    -------
    List[float]
        Parsed list of valid multiplier scores.

    Raises
    ------
    ValueError
        If no valid numbers are found in the text.
    """
    # Replace common delimiters with spaces, then split
    cleaned = raw_text.replace(",", " ").replace("\n", " ").replace("\r", " ")
    tokens = cleaned.split()

    scores: List[float] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Normalise: remove trailing/leading 'x' or 'X' (common in multiplier
        # notation like "1.00x") so the numeric core is preserved for conversion.
        clean_token = token.lower().replace("x", "")

        try:
            val = float(clean_token)
            if val <= 0:
                raise ValueError(f"Score must be positive: {token}")
            scores.append(val)
        except ValueError as e:
            if "Score must be positive" in str(e):
                raise
            raise ValueError(
                f"Invalid number found: '{token}'. Please ensure all entries "
                f"are numeric values (e.g., 1.23, 5.40)."
            )

    if not scores:
        raise ValueError(
            "No valid scores found. Please paste a sequence of numbers "
            "separated by commas, spaces, or newlines."
        )

    return scores


# ------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------

class ScoreAnalyzerApp:
    """Main application window for the Aviator Score Analyzer."""

    def __init__(self) -> None:
        self.root = ctk.CTk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(720, 780)

        # Center the window on screen
        self._center_window()

        # State for image tab
        self._current_pil_image: Optional[Image.Image] = None
        self._current_photo: Optional[ImageTk.PhotoImage] = None

        # Ensure database is ready
        setup_db()

        # --- Build UI ---
        self._build_widgets()
        self._refresh_recent_scores()

    def _center_window(self) -> None:
        """Center the window on the user's screen."""
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - WINDOW_WIDTH) // 2
        y = (screen_h - WINDOW_HEIGHT) // 2
        self.root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        """Create and arrange all UI widgets."""
        # Main window: title fixed, tabview fills space, prediction fixed,
        # recent scores fills remaining space
        self.root.grid_rowconfigure(0, weight=0)  # Title
        self.root.grid_rowconfigure(1, weight=1)  # Tabview (fills space)
        self.root.grid_rowconfigure(2, weight=0)  # Shared prediction output
        self.root.grid_rowconfigure(3, weight=1)  # Shared recent scores
        self.root.grid_columnconfigure(0, weight=1)

        # -- Title Section --
        self._build_title_section()

        # -- Tabbed Interface (no fixed height — fills via weight=1) --
        self.tabview = ctk.CTkTabview(
            self.root,
            corner_radius=CORNER_RADIUS,
        )
        self.tabview.grid(row=1, column=0, **PAD, sticky="nsew")

        self.tab_manual = self.tabview.add("  ⌨️  Manual Input  ")
        self.tab_image = self.tabview.add("  📷  Image / Screenshot  ")

        # Each tab gets a scrollable frame so content is never clipped
        self._build_manual_tab()
        self._build_image_tab()

        # -- Shared Prediction Output --
        self._build_prediction_section(row=2)

        # -- Shared Recent Scores --
        self._build_recent_scores_section(row=3)

    # ================ Title Section ================

    def _build_title_section(self) -> None:
        """Build the header/title area with accent styling."""
        title_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=(16, 4), sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)

        # Main title
        title_label = ctk.CTkLabel(
            title_frame,
            text="📊  Aviator Score Analyzer",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title_label.grid(row=0, column=0, pady=(0, 2))

        # Subtitle
        subtitle = ctk.CTkLabel(
            title_frame,
            text="Predict the next multiplier from manual input or OCR-extracted scores",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        subtitle.grid(row=1, column=0)

        # Accent bar below title
        accent_frame = ctk.CTkFrame(
            title_frame,
            height=3,
            fg_color=COLOR_ACCENT,
            corner_radius=2,
        )
        accent_frame.grid(row=2, column=0, pady=(10, 0), sticky="ew")
        # Prevent the accent bar from expanding vertically
        accent_frame.grid_propagate(False)

    # ================ Scrollable Tab Wrapper ================

    def _make_scrollable_tab(self, tab: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
        """
        Wrap a tab's contents inside a CTkScrollableFrame so all widgets
        remain accessible even when the window is too small.
        """
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        scroll_frame = ctk.CTkScrollableFrame(
            tab,
            corner_radius=0,
            border_width=0,
            fg_color="transparent",
            scrollbar_fg_color=("gray90", "#333333"),
            scrollbar_button_color=("gray70", "#555555"),
            scrollbar_button_hover_color=("gray50", "#777777"),
        )
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(0, weight=1)

        return scroll_frame

    # ================ Manual Input Tab ================

    def _build_manual_tab(self) -> None:
        tab = self.tab_manual
        scroll = self._make_scrollable_tab(tab)

        # -- Section label --
        section_label = ctk.CTkLabel(
            scroll,
            text="✏️  Enter Scores Manually",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        section_label.grid(row=0, column=0, padx=6, pady=(12, 4), sticky="w")

        # -- Input text area --
        self.input_text = ctk.CTkTextbox(
            scroll,
            height=130,
            font=ctk.CTkFont(size=14),
            corner_radius=CORNER_RADIUS_SMALL,
            border_width=1,
        )
        self._input_placeholder = (
            "Paste scores here... e.g.: 1.23, 5.40, 2.10, 1.05, 12.30\n"
            "Or separated by spaces / newlines"
        )
        self.input_text.insert("1.0", self._input_placeholder)
        self.input_text.bind(
            "<FocusIn>",
            lambda e: self._clear_placeholder(self.input_text, self._input_placeholder),
        )
        self.input_text.bind(
            "<FocusOut>",
            lambda e: self._restore_placeholder(self.input_text, self._input_placeholder),
        )
        self.input_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=(2, 8))

        # -- Buttons --
        btn_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=6, pady=(4, 12), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self.process_btn = ctk.CTkButton(
            btn_frame,
            text="🚀  Process & Predict",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=BUTTON_HEIGHT,
            corner_radius=CORNER_RADIUS_SMALL,
            fg_color=COLOR_PROCESS,
            hover_color=COLOR_PROCESS_HOVER,
            command=self._on_process_predict,
        )
        self.process_btn.grid(row=0, column=0, padx=(0, 6), pady=4, sticky="ew")

        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="🗑️  Clear Database",
            font=ctk.CTkFont(size=14),
            height=BUTTON_HEIGHT,
            corner_radius=CORNER_RADIUS_SMALL,
            fg_color=COLOR_DANGER,
            hover_color=COLOR_DANGER_HOVER,
            command=self._on_clear_database,
        )
        self.clear_btn.grid(row=0, column=1, padx=(6, 0), pady=4, sticky="ew")

    # ================ Image / Screenshot Tab ================

    def _build_image_tab(self) -> None:
        tab = self.tab_image
        scroll = self._make_scrollable_tab(tab)

        # -- Section label --
        section_label = ctk.CTkLabel(
            scroll,
            text="🖼️  Load an Image for OCR Extraction",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        section_label.grid(row=0, column=0, padx=6, pady=(12, 4), sticky="w")

        # -- Action buttons --
        action_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        action_frame.grid(row=1, column=0, padx=6, pady=(2, 6), sticky="ew")
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)

        self.clipboard_btn = ctk.CTkButton(
            action_frame,
            text="📋  Paste from Clipboard",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=BUTTON_HEIGHT_SMALL,
            corner_radius=CORNER_RADIUS_SMALL,
            fg_color=COLOR_INFO,
            hover_color=COLOR_INFO_HOVER,
            command=self._on_paste_clipboard,
        )
        self.clipboard_btn.grid(row=0, column=0, padx=(0, 5), pady=4, sticky="ew")

        self.browse_btn = ctk.CTkButton(
            action_frame,
            text="📁  Browse Image File",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=BUTTON_HEIGHT_SMALL,
            corner_radius=CORNER_RADIUS_SMALL,
            fg_color=COLOR_INFO,
            hover_color=COLOR_INFO_HOVER,
            command=self._on_browse_image,
        )
        self.browse_btn.grid(row=0, column=1, padx=(5, 0), pady=4, sticky="ew")

        # -- Image preview card --
        preview_card = ctk.CTkFrame(
            scroll,
            corner_radius=CORNER_RADIUS_SMALL,
            border_width=1,
        )
        preview_card.grid(row=2, column=0, padx=6, pady=6, sticky="ew")
        preview_card.grid_columnconfigure(0, weight=1)

        preview_header = ctk.CTkLabel(
            preview_card,
            text="Image Preview",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )
        preview_header.grid(row=0, column=0, padx=10, pady=(8, 0), sticky="w")

        self.preview_label = ctk.CTkLabel(
            preview_card,
            text="No image loaded.\nPaste from clipboard or browse a file.",
            font=ctk.CTkFont(size=13),
            fg_color=("#e8e8e8", "#2b2b2b"),
            corner_radius=CORNER_RADIUS_SMALL,
            height=160,
        )
        self.preview_label.grid(
            row=1, column=0, padx=10, pady=(6, 10), sticky="ew"
        )

        # -- Extract & Predict button --
        self.extract_btn = ctk.CTkButton(
            scroll,
            text="🔍  Extract Scores & Predict",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=BUTTON_HEIGHT,
            corner_radius=CORNER_RADIUS_SMALL,
            fg_color=COLOR_EXTRACT,
            hover_color=COLOR_EXTRACT_HOVER,
            state="disabled",  # enabled only when an image is loaded
            command=self._on_extract_predict,
        )
        self.extract_btn.grid(row=3, column=0, padx=6, pady=(8, 4), sticky="ew")

        # -- Clear Database button --
        self.clear_img_btn = ctk.CTkButton(
            scroll,
            text="🗑️  Clear Database",
            font=ctk.CTkFont(size=14),
            height=BUTTON_HEIGHT,
            corner_radius=CORNER_RADIUS_SMALL,
            fg_color=COLOR_DANGER,
            hover_color=COLOR_DANGER_HOVER,
            command=self._on_clear_database,
        )
        self.clear_img_btn.grid(row=4, column=0, padx=6, pady=(4, 12), sticky="ew")

    # ================ Shared Output Sections ================

    def _build_prediction_section(self, row: int) -> None:
        """Prediction result and training status (shared across tabs)."""
        output_frame = ctk.CTkFrame(
            self.root,
            corner_radius=CORNER_RADIUS,
            border_width=1,
        )
        output_frame.grid(row=row, column=0, **PAD, sticky="nsew")
        output_frame.grid_columnconfigure(0, weight=1)

        # Prediction value - large and prominent
        self.prediction_label = ctk.CTkLabel(
            output_frame,
            text="Predicted Next Score:  ---",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLOR_PREDICTION,
        )
        self.prediction_label.grid(row=0, column=0, padx=14, pady=(14, 4))

        # Status line
        self.status_label = ctk.CTkLabel(
            output_frame,
            text="Status: Waiting for input...",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        self.status_label.grid(row=1, column=0, padx=14, pady=(0, 12))

    def _build_recent_scores_section(self, row: int) -> None:
        """Recent scores listbox (shared across tabs)."""
        recent_frame = ctk.CTkFrame(
            self.root,
            corner_radius=CORNER_RADIUS,
            border_width=1,
        )
        recent_frame.grid(row=row, column=0, **PAD, sticky="nsew")
        recent_frame.grid_columnconfigure(0, weight=1)
        recent_frame.grid_rowconfigure(1, weight=1)

        # Header row with icon and count
        header_frame = ctk.CTkFrame(recent_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=14, pady=(10, 2), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        recent_header = ctk.CTkLabel(
            header_frame,
            text="📋  Recent Scores",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        recent_header.grid(row=0, column=0, sticky="w")

        self.count_label = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="e",
        )
        self.count_label.grid(row=0, column=1, sticky="e")

        # Determine initial listbox colors based on current appearance mode
        self._update_listbox_colors()

        self.recent_listbox = tk.Listbox(
            recent_frame,
            font=("Consolas", 12),
            height=6,
            bg=_listbox_bg,
            fg=_listbox_fg,
            selectbackground=COLOR_ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightcolor="#cccccc",
            highlightbackground="#555555",
        )
        self.recent_listbox.grid(
            row=1, column=0, padx=14, pady=(4, 12), sticky="nsew"
        )

        # Add scrollbar
        scrollbar = ctk.CTkScrollbar(
            recent_frame, command=self.recent_listbox.yview
        )
        scrollbar.grid(row=1, column=1, pady=(4, 12), padx=(0, 14), sticky="ns")
        self.recent_listbox.configure(yscrollcommand=scrollbar.set)

    def _update_listbox_colors(self) -> None:
        """Set listbox colors based on current CustomTkinter theme mode."""
        global _listbox_bg, _listbox_fg
        if ctk.get_appearance_mode() == "Light":
            _listbox_bg = "#ffffff"
            _listbox_fg = "#1a1a1a"
        else:
            _listbox_bg = "#2b2b2b"
            _listbox_fg = "#e0e0e0"

    # ------------------------------------------------------------------
    # Placeholder Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_placeholder(text_widget: ctk.CTkTextbox, placeholder: str) -> None:
        """Remove placeholder text when the widget gains focus."""
        current = text_widget.get("1.0", tk.END).strip()
        if current == placeholder.strip():
            text_widget.delete("1.0", tk.END)

    @staticmethod
    def _restore_placeholder(text_widget: ctk.CTkTextbox, placeholder: str) -> None:
        """Restore placeholder text if the widget is empty on focus loss."""
        current = text_widget.get("1.0", tk.END).strip()
        if not current:
            text_widget.insert("1.0", placeholder)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _display_image_thumbnail(self, pil_image: Image.Image) -> None:
        """Resize and display a PIL image as a thumbnail in the preview area."""
        # Resize maintaining aspect ratio
        thumb = pil_image.copy()
        thumb.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

        self._current_photo = ImageTk.PhotoImage(thumb)
        self.preview_label.configure(
            image=self._current_photo,
            text="",
            fg_color=("white", "black"),
        )
        self.extract_btn.configure(state="normal")

    def _clear_image_preview(self) -> None:
        """Reset the image preview to the placeholder state."""
        self._current_pil_image = None
        self._current_photo = None
        self.preview_label.configure(
            image=None,
            text="No image loaded.\nPaste from clipboard or browse a file.",
            fg_color=("#e8e8e8", "#2b2b2b"),
        )
        self.extract_btn.configure(state="disabled")

    # ------------------------------------------------------------------
    # Shared prediction / training pipeline
    # ------------------------------------------------------------------

    def _run_prediction_pipeline(self, scores: List[float]) -> None:
        """
        Common pipeline after scores have been obtained (either from manual
        text or OCR): insert into DB, train model, update UI.

        Parameters
        ----------
        scores : List[float]
            The parsed/extracted multiplier scores.
        """
        if not scores:
            return

        # Combine with existing history for training context
        all_scores = fetch_all_scores()

        # Train & predict
        try:
            prediction, mse, total_samples = train_and_predict_from_history(
                all_scores
            )
        except ValueError as e:
            messagebox.showerror("Training Error", str(e))
            self.status_label.configure(
                text=f"Status: ⚠️  {e}",
                text_color="orange",
            )
            self._refresh_recent_scores()
            return

        # Update UI
        self.prediction_label.configure(
            text=f"Predicted Next Score:  {prediction:.2f}x"
        )

        mse_text = ""
        if mse is not None:
            mse_text = f"  |  MSE: {mse:.6f}"
        self.status_label.configure(
            text=(
                f"Status: ✅  Model trained on {total_samples} samples"
                f"{mse_text}"
            ),
            text_color=COLOR_SUCCESS,
        )

        self._refresh_recent_scores()

    # ------------------------------------------------------------------
    # Event Handlers - Manual Tab
    # ------------------------------------------------------------------

    def _on_process_predict(self) -> None:
        """
        Handle the 'Process & Predict' button click (Manual tab):
          1. Parse input text.
          2. Save scores to database.
          3. Fetch all history.
          4. Train XGBoost model & predict.
          5. Update UI with results.
        """
        raw_text = self.input_text.get("1.0", tk.END).strip()

        # If the user hasn't replaced the placeholder text, treat as empty
        if raw_text == self._input_placeholder.strip():
            raw_text = ""

        if not raw_text:
            messagebox.showwarning(
                "No Input", "Please paste some scores before processing."
            )
            return

        # --- Step 1: Parse ---
        try:
            scores = parse_score_text(raw_text)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        # --- Step 2: Save to database ---
        insert_scores(raw_text, scores)

        # --- Step 3-5: Shared pipeline ---
        self._run_prediction_pipeline(scores)

        # Clear input after successful processing
        self.input_text.delete("1.0", tk.END)

    # ------------------------------------------------------------------
    # Event Handlers - Image Tab
    # ------------------------------------------------------------------

    def _on_paste_clipboard(self) -> None:
        """Grab an image from the clipboard and display a preview."""
        try:
            from PIL import ImageGrab
            pil_image = ImageGrab.grabclipboard()
        except Exception:
            pil_image = None

        if pil_image is None:
            messagebox.showwarning(
                "Clipboard Empty",
                "No image found in clipboard!",
            )
            return

        # Check if it's actually an Image instance (grabclipboard can return
        # a list of file paths on some systems)
        if not isinstance(pil_image, Image.Image):
            messagebox.showwarning(
                "Clipboard Empty",
                "No image found in clipboard!",
            )
            return

        self._current_pil_image = pil_image
        self._display_image_thumbnail(pil_image)

    def _on_browse_image(self) -> None:
        """Open a file dialog to select an image file and display a preview."""
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            pil_image = Image.open(file_path).copy()
        except Exception as e:
            messagebox.showerror(
                "File Error",
                f"Could not open the selected file:\n{e}",
            )
            return

        self._current_pil_image = pil_image
        self._display_image_thumbnail(pil_image)

    def _on_extract_predict(self) -> None:
        """
        Extract scores from the currently loaded image via OCR, save to DB,
        then run the shared training/prediction pipeline.
        """
        if self._current_pil_image is None:
            messagebox.showwarning(
                "No Image",
                "Please paste an image from the clipboard or browse a file first.",
            )
            return

        # --- OCR extraction ---
        try:
            scores = extract_scores_from_image(self._current_pil_image)
        except ValueError as e:
            messagebox.showwarning(
                "OCR Failed",
                str(e),
            )
            return
        except Exception as e:
            messagebox.showerror(
                "OCR Error",
                f"An unexpected error occurred during OCR:\n{e}",
            )
            return

        if not scores:
            messagebox.showwarning(
                "OCR Failed",
                "Could not detect any valid scores in the image. "
                "Please ensure the numbers are clearly visible.",
            )
            return

        # --- Save to database ---
        # Use a descriptive raw_text placeholder for OCR-sourced entries
        raw_label = f"[OCR extracted: {len(scores)} scores]"
        insert_scores(raw_label, scores)

        # --- Shared prediction pipeline ---
        self._run_prediction_pipeline(scores)

        # Clear the image preview after successful extraction
        self._clear_image_preview()

    # ------------------------------------------------------------------
    # Event Handlers - Shared
    # ------------------------------------------------------------------

    def _on_clear_database(self) -> None:
        """
        Handle the 'Clear Database' button click.
        Shows a confirmation dialog, then wipes all data.
        """
        confirm = messagebox.askyesno(
            "Confirm Clear",
            "Are you sure you want to delete ALL historical scores?\n"
            "This action cannot be undone.",
        )
        if confirm:
            clear_database()
            self.prediction_label.configure(
                text="Predicted Next Score:  ---"
            )
            self.status_label.configure(
                text="Status: 🗑️  Database cleared. Waiting for input...",
                text_color="gray",
            )
            self._refresh_recent_scores()
            messagebox.showinfo(
                "Cleared", "All game history has been deleted."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_recent_scores(self) -> None:
        """Reload the recent scores listbox from the database."""
        self.recent_listbox.delete(0, tk.END)

        recent = fetch_recent_scores(limit=10)
        total = get_total_count()

        # Update the count label
        if total > 0:
            self.count_label.configure(text=f"Total: {total}  |  Showing last {len(recent)}")
        else:
            self.count_label.configure(text="")

        if not recent:
            self.recent_listbox.insert(
                tk.END, "  No scores in database yet."
            )
            return

        # Display in reverse chronological order (most recent first)
        for record_id, score, ts in recent:
            display_text = f"  #{record_id:>4}  {score:>8.2f}x    {ts}"
            self.recent_listbox.insert(tk.END, display_text)

    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()


def create_gui() -> None:
    """Convenience function to create and run the application."""
    app = ScoreAnalyzerApp()
    app.run()


if __name__ == "__main__":
    create_gui()