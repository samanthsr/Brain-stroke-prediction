# ============================================================
# Brain Stroke Prediction — Premium UI (Reference Style)
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import math
from PIL import Image, ImageTk
# ── Load Models ─────────────────────────────────────────────
try:
    with open('stroke_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    nn_model = load_model('stroke_nn_model.keras')
    with open('stroke_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('stroke_labels.pkl', 'rb') as f:
        stroke_labels = pickle.load(f)
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── Colors ───────────────────────────────────────────────────
BG_MAIN     = '#0a0e1a'
BG_SIDEBAR  = '#0d1120'
BG_CARD     = '#111827'
BG_CARD2    = '#0f1929'
BG_INPUT    = '#1a2035'
BG_HEADER   = '#0d1120'
ACCENT      = '#4f8ef7'
ACCENT2     = '#1a3a6e'
TEXT_PRI    = '#ffffff'
TEXT_SEC    = '#8899bb'
TEXT_HINT   = '#445577'
BORDER      = '#1e2d4a'
GREEN       = '#2ecc71'
RED         = '#e74c3c'
ORANGE      = '#f39c12'
CYAN        = '#00d4ff'

STROKE_COLORS = {
    'No Stroke'  : '#2ecc71',
    'Ischemic'   : '#2ecc71',
    'Hemorrhagic': '#e74c3c',
    'TIA'        : '#f39c12'
}

BAR_COLORS = {
    'No Stroke'  : '#2ecc71',
    'Ischemic'   : '#2ecc71',
    'Hemorrhagic': '#e74c3c',
    'TIA'        : '#f39c12'
}

class BrainImageCard(tk.Frame):
    """Show real brain MRI image with dynamic highlight overlay"""
    def __init__(self, parent, stroke_type, probability,
                 color, **kwargs):
        super().__init__(parent, bg=BG_CARD2, **kwargs)

        image_map = {
            'Ischemic'   : 'ischemic_brain.png',
            'Hemorrhagic': 'hemorrhagic_brain.png',
            'TIA'        : 'tia_brain.png'
        }

        img_path = image_map.get(stroke_type, None)
        img_size = (180, 160)

        canvas = tk.Canvas(self, width=img_size[0],
                           height=img_size[1],
                           bg=BG_CARD2,
                           highlightthickness=0)
        canvas.pack(pady=4)

        try:
            from PIL import Image, ImageTk, ImageDraw, ImageFilter
            import colorsys

            # Load brain image
            img = Image.open(img_path).convert('RGBA')
            img = img.resize(img_size, Image.LANCZOS)

            # Create overlay based on stroke type & probability
            overlay = Image.new('RGBA', img_size,
                                (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Alpha based on probability (more prob = more visible)
            alpha = int(180 * probability)
            alpha = max(60, min(220, alpha))

            # Parse color to RGB
            col = color.lstrip('#')
            r = int(col[0:2], 16)
            g = int(col[2:4], 16)
            b = int(col[4:6], 16)

            cx = img_size[0] // 2
            cy = img_size[1] // 2

            if stroke_type == 'Ischemic':
                # Left hemisphere region — size based on prob
                region_w = int(55 * probability) + 20
                region_h = int(45 * probability) + 15
                x1 = cx - region_w - 10
                y1 = cy - region_h // 2
                x2 = cx - 10
                y2 = cy + region_h // 2
                draw.ellipse([x1, y1, x2, y2],
                             fill=(r, g, b, alpha))
                # Inner darker core
                draw.ellipse([x1+10, y1+10,
                              x2-10, y2-10],
                             fill=(r, g, b,
                                   min(255, alpha+40)))

            elif stroke_type == 'Hemorrhagic':
                # Center bleed — size based on prob
                radius = int(35 * probability) + 10
                draw.ellipse([cx-radius, cy-radius,
                              cx+radius, cy+radius],
                             fill=(r, g, b, alpha))
                # Bright center
                inner_r = max(5, radius // 2)
                draw.ellipse([cx-inner_r, cy-inner_r,
                              cx+inner_r, cy+inner_r],
                             fill=(255, 50, 50,
                                   min(255, alpha+50)))

            elif stroke_type == 'TIA':
                # Multiple small spots — count based on prob
                num_spots = max(2, int(6 * probability))
                import math
                for i in range(num_spots):
                    angle = (360 / num_spots) * i
                    rad   = math.radians(angle)
                    dist  = int(35 * probability) + 15
                    sx = cx + int(dist * math.cos(rad))
                    sy = cy + int(dist * math.sin(rad))
                    spot_r = max(4, int(10 * probability))
                    draw.ellipse([sx-spot_r, sy-spot_r,
                                  sx+spot_r, sy+spot_r],
                                 fill=(r, g, b, alpha))

            # Blur overlay for realistic look
            overlay = overlay.filter(
                ImageFilter.GaussianBlur(radius=4))

            # Combine brain + overlay
            combined = Image.alpha_composite(img, overlay)
            combined = combined.convert('RGB')

            self.photo = ImageTk.PhotoImage(combined)
            canvas.create_image(0, 0, anchor='nw',
                                image=self.photo)

            # Probability border glow at bottom
            canvas.create_rectangle(
                0, img_size[1]-5,
                int(img_size[0] * probability),
                img_size[1],
                fill=color, outline='')

        except Exception as e:
            print(f"Image load error for {stroke_type}: {e}")
            # Fallback if no image found
            canvas.create_rectangle(0, 0,
                img_size[0], img_size[1],
                fill='#1a1a2e', outline='')

            # Draw fallback brain circle
            cx = img_size[0] // 2
            cy = img_size[1] // 2
            r  = 65

            canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                               fill='#1a1a2e',
                               outline='#334466', width=2)

            # Highlight region size based on probability
            region = int(r * probability) + 10

            if stroke_type == 'Ischemic':
                canvas.create_oval(
                    cx-region-10, cy-region//2,
                    cx-10, cy+region//2,
                    fill=color, outline='')
            elif stroke_type == 'Hemorrhagic':
                canvas.create_oval(
                    cx-region//2, cy-region//2,
                    cx+region//2, cy+region//2,
                    fill=color, outline='')
            elif stroke_type == 'TIA':
                import math
                num = max(2, int(5 * probability))
                for i in range(num):
                    angle = (360/num) * i
                    rad   = math.radians(angle)
                    sx = cx + int(40*math.cos(rad))
                    sy = cy + int(40*math.sin(rad))
                    sr = max(4, int(10*probability))
                    canvas.create_oval(
                        sx-sr, sy-sr, sx+sr, sy+sr,
                        fill=color, outline='')

            canvas.create_text(
                cx, img_size[1]-15,
                text=f'{probability*100:.1f}% affected',
                fill=color,
                font=('Helvetica', 8))

class StrokeApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Machine Learning Based Diagnosis & Prediction "
            "of Three Types of Brain Stroke")
        self.root.geometry("1200x720")
        self.root.configure(bg=BG_MAIN)
        self.root.resizable(True, True)
        self.current_page = tk.StringVar(value='patient')
        self.build_ui()
        if not models_loaded:
            messagebox.showerror("Model Error",
                f"Run stroke_advanced.py first!\n{load_error}")

    # ── Build UI ─────────────────────────────────────────────
    def build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.build_header()
        self.build_main()
        self.build_sidebar()

    # ── Header ───────────────────────────────────────────────
    def build_header(self):
        header = tk.Frame(self.root, bg=BG_HEADER, height=70)
        header.grid(row=0, column=0, columnspan=2,
                    sticky='ew')
        header.columnconfigure(1, weight=1)
        header.grid_propagate(False)

        # Brain icon + title
        left = tk.Frame(header, bg=BG_HEADER)
        left.grid(row=0, column=0, padx=20, pady=12,
                  sticky='w')

        icon_canvas = tk.Canvas(left, width=50, height=50,
                                bg=BG_HEADER,
                                highlightthickness=0)
        icon_canvas.pack(side='left')
        icon_canvas.create_oval(2, 2, 48, 48,
                                fill='#1a2a4a',
                                outline=ACCENT, width=2)
        icon_canvas.create_text(25, 25, text='🧠',
                                font=('Helvetica', 22))

        title_frame = tk.Frame(left, bg=BG_HEADER)
        title_frame.pack(side='left', padx=12)
        tk.Label(title_frame,
                 text='Machine Learning Based Diagnosis & Prediction',
                 bg=BG_HEADER, fg=TEXT_PRI,
                 font=('Helvetica', 13, 'bold')).pack(anchor='w')
        tk.Label(title_frame,
                 text='of Three Types of Brain Stroke',
                 bg=BG_HEADER, fg=TEXT_PRI,
                 font=('Helvetica', 13, 'bold')).pack(anchor='w')

        # Date time + home
        right = tk.Frame(header, bg=BG_HEADER)
        right.grid(row=0, column=1, padx=20, sticky='e')

        now = datetime.now()
        self.time_label = tk.Label(right,
            text=now.strftime('%d %b %Y  |  %I:%M %p'),
            bg=BG_HEADER, fg=TEXT_SEC,
            font=('Helvetica', 11))
        self.time_label.pack(side='left', padx=(0,20))
        self.update_time()

        tk.Button(right, text='⌂  Home',
                  command=lambda: self.show_page('patient'),
                  bg=BG_HEADER, fg=TEXT_SEC,
                  font=('Helvetica', 11),
                  relief='flat', bd=0,
                  cursor='hand2',
                  activebackground=BG_HEADER,
                  activeforeground=TEXT_PRI).pack(side='left')

        # Separator
        tk.Frame(self.root, bg=BORDER, height=1).grid(
            row=0, column=0, columnspan=2,
            sticky='sew')

    def update_time(self):
        now = datetime.now()
        self.time_label.config(
            text=now.strftime('%d %b %Y  |  %I:%M %p'))
        self.root.after(60000, self.update_time)

    # ── Sidebar ──────────────────────────────────────────────
    def build_sidebar(self):
        sidebar = tk.Frame(self.root, bg=BG_SIDEBAR, width=220)
        sidebar.grid(row=1, column=0, sticky='nsew')
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        menu_items = [
            ('patient',    '👤  Patient Information'),
            ('result',     '📊  Prediction Result'),
            ('about',      'ℹ️   About System'),
        ]

        self.menu_buttons = {}
        for i, (page, label) in enumerate(menu_items):
            btn = tk.Button(sidebar, text=label,
                            command=lambda p=page: self.show_page(p),
                            bg=BG_SIDEBAR, fg=TEXT_SEC,
                            font=('Helvetica', 11),
                            relief='flat', bd=0,
                            anchor='w', padx=20, pady=14,
                            cursor='hand2',
                            activebackground=ACCENT2,
                            activeforeground=TEXT_PRI)
            btn.grid(row=i, column=0, sticky='ew',
                     pady=(2,0))
            self.menu_buttons[page] = btn

        # Warning at bottom
        warn = tk.Frame(sidebar, bg=BG_SIDEBAR)
        warn.grid(row=10, column=0, sticky='sew',
                  padx=15, pady=20)
        sidebar.rowconfigure(10, weight=1)
        tk.Label(warn,
                 text='🛡️',
                 bg=BG_SIDEBAR, fg=TEXT_HINT,
                 font=('Helvetica', 16)).pack(anchor='w')
        tk.Label(warn,
                 text='This system is for educational\n'
                      'and research purposes only.\n'
                      'Not a substitute for professional\n'
                      'medical diagnosis.',
                 bg=BG_SIDEBAR, fg=TEXT_HINT,
                 font=('Helvetica', 9),
                 justify='left',
                 wraplength=180).pack(anchor='w', pady=(6,0))

        self.sidebar = sidebar
        self.show_page('patient')

    def show_page(self, page):
        for name, btn in self.menu_buttons.items():
            if name == page:
                btn.config(bg=ACCENT2, fg=TEXT_PRI)
            else:
                btn.config(bg=BG_SIDEBAR, fg=TEXT_SEC)
        self.current_page.set(page)
        if page == 'patient':
            self.patient_frame.tkraise()
        elif page == 'result':
            self.result_frame.tkraise()
        elif page == 'about':
            self.about_frame.tkraise()

    # ── Main Content ─────────────────────────────────────────
    def build_main(self):
        main = tk.Frame(self.root, bg=BG_MAIN)
        main.grid(row=1, column=1, sticky='nsew',
                  padx=0, pady=0)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # Stack pages
        self.patient_frame = tk.Frame(main, bg=BG_MAIN)
        self.result_frame  = tk.Frame(main, bg=BG_MAIN)
        self.about_frame   = tk.Frame(main, bg=BG_MAIN)

        for frame in (self.patient_frame,
                      self.result_frame,
                      self.about_frame):
            frame.grid(row=0, column=0, sticky='nsew')

        self.build_patient_page()
        self.build_result_page()
        self.build_about_page()

    # ── Patient Page ─────────────────────────────────────────
    def build_patient_page(self):
        f = self.patient_frame
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        # Title
        tk.Label(f, text='Patient Information',
                 bg=BG_MAIN, fg=TEXT_PRI,
                 font=('Helvetica', 16, 'bold')).grid(
            row=0, column=0, columnspan=2,
            sticky='w', padx=30, pady=(20,15))

        # Card
        card = tk.Frame(f, bg=BG_CARD,
                        highlightbackground=BORDER,
                        highlightthickness=1)
        card.grid(row=1, column=0, columnspan=2,
                  sticky='nsew', padx=30, pady=(0,10))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

        entry_s = {'bg': BG_INPUT, 'fg': TEXT_PRI,
                   'font': ('Helvetica', 11),
                   'relief': 'flat', 'bd': 6,
                   'insertbackground': TEXT_PRI}

        def field(parent, label, var_or_widget,
                  row, col, is_combo=False,
                  values=None):
            fr = tk.Frame(parent, bg=BG_CARD)
            fr.grid(row=row, column=col, sticky='ew',
                    padx=20, pady=8)
            fr.columnconfigure(0, weight=1)
            tk.Label(fr, text=label, bg=BG_CARD,
                     fg=TEXT_SEC,
                     font=('Helvetica', 9)).pack(anchor='w')
            wrap = tk.Frame(fr, bg=BG_INPUT,
                            highlightbackground=BORDER,
                            highlightthickness=1)
            wrap.pack(fill='x', pady=(3,0))
            if is_combo:
                w = ttk.Combobox(wrap,
                                 textvariable=var_or_widget,
                                 values=values,
                                 state='readonly',
                                 font=('Helvetica', 11))
            else:
                w = tk.Entry(wrap,
                             textvariable=var_or_widget,
                             **entry_s)
            w.pack(fill='x', padx=2, pady=2)

        self.gender_var  = tk.StringVar(value='Male')
        self.age_var     = tk.StringVar(value='67')
        self.hyp_var     = tk.StringVar(value='Yes')
        self.hd_var      = tk.StringVar(value='Yes')
        self.married_var = tk.StringVar(value='Yes')
        self.work_var    = tk.StringVar(value='Private')
        self.res_var     = tk.StringVar(value='Urban')
        self.glucose_var = tk.StringVar(value='185.0')
        self.bmi_var     = tk.StringVar(value='34.5')
        self.smoke_var   = tk.StringVar(value='formerly smoked')

        # Row 0
        field(card, 'Gender', self.gender_var, 0, 0,
              True, ['Male','Female'])
        field(card, 'Age', self.age_var, 0, 1)
        # Row 1
        field(card, 'Hypertension', self.hyp_var, 1, 0,
              True, ['No','Yes'])
        field(card, 'Heart Disease', self.hd_var, 1, 1,
              True, ['No','Yes'])
        # Row 2
        field(card, 'Ever Married', self.married_var, 2, 0,
              True, ['Yes','No'])
        field(card, 'Residence Type', self.res_var, 2, 1,
              True, ['Urban','Rural'])
        # Row 3
        field(card, 'Avg Glucose Level (mg/dL)',
              self.glucose_var, 3, 0)
        field(card, 'BMI', self.bmi_var, 3, 1)
        # Row 4
        field(card, 'Work Type', self.work_var, 4, 0,
              True, ['Private','Govt','Self-employed',
                     'Children','Never worked'])
        field(card, 'Smoking Status', self.smoke_var, 4, 1,
              True, ['never smoked','formerly smoked',
                     'smokes','Unknown'])

        # Predict button
        btn_row = tk.Frame(f, bg=BG_MAIN)
        btn_row.grid(row=2, column=0, columnspan=2,
                     pady=15, padx=30, sticky='ew')
        btn_row.columnconfigure(0, weight=1)

        tk.Button(btn_row,
                  text='🔍   ANALYZE & PREDICT STROKE TYPE',
                  command=self.predict,
                  bg=ACCENT, fg='white',
                  font=('Helvetica', 13, 'bold'),
                  relief='flat', bd=0, pady=13,
                  cursor='hand2',
                  activebackground='#3a7af0',
                  activeforeground='white').grid(
            row=0, column=0, sticky='ew', padx=(0,10))

        tk.Button(btn_row, text='🔄  Clear',
                  command=self.clear_form,
                  bg=BG_CARD, fg=TEXT_SEC,
                  font=('Helvetica', 13),
                  relief='flat', bd=0, pady=13,
                  cursor='hand2').grid(
            row=0, column=1, padx=(0,0), ipadx=20)

    # ── Result Page ──────────────────────────────────────────
    def build_result_page(self):
        f = self.result_frame
        f.columnconfigure(0, weight=1)

        # Title bar
        title_bar = tk.Frame(f, bg=BG_MAIN)
        title_bar.grid(row=0, column=0, sticky='ew',
                       padx=30, pady=(20,10))
        tk.Label(title_bar, text='PREDICTION RESULT',
         bg=BG_MAIN, fg=CYAN,
         font=('Helvetica', 14, 'bold')).pack(anchor='w')

        # Main diagnosis card
        self.diag_card = tk.Frame(f, bg=BG_CARD2,
                                  highlightbackground=BORDER,
                                  highlightthickness=1)
        self.diag_card.grid(row=1, column=0, sticky='ew',
                            padx=30, pady=(0,12))
        self.diag_card.columnconfigure(0, weight=1)

        self.diag_icon = tk.Label(self.diag_card, text='◯',
                                  bg=BG_CARD2, fg=TEXT_HINT,
                                  font=('Helvetica', 36))
        self.diag_icon.grid(row=0, column=0,
                            padx=20, pady=20, sticky='w')

        self.diag_sub = tk.Label(self.diag_card,
                                 text='DIAGNOSIS',
                                 bg=BG_CARD2, fg=TEXT_HINT,
                                 font=('Helvetica', 10, 'bold'))
        self.diag_sub.grid(row=0, column=1, sticky='sw',
                           pady=(20,2))

        self.diag_label = tk.Label(self.diag_card,
                                   text='Run prediction first...',
                                   bg=BG_CARD2, fg=TEXT_HINT,
                                   font=('Helvetica', 22, 'bold'))
        self.diag_label.grid(row=0, column=1, sticky='nw',
                             pady=(2,20))

        self.conf_label = tk.Label(self.diag_card,
                                   text='Confidence Score\n—',
                                   bg=BG_CARD2, fg=TEXT_HINT,
                                   font=('Helvetica', 12),
                                   justify='right')
        self.conf_label.grid(row=0, column=2, sticky='e',
                             padx=30, pady=20)

        self.diag_card.columnconfigure(1, weight=1)

        # Probabilities label
        self.prob_title = tk.Label(f,
            text='PREDICTION PROBABILITIES (Three Types of Brain Stroke)',
            bg=BG_MAIN, fg=TEXT_SEC,
            font=('Helvetica', 10, 'bold'))
        self.prob_title.grid(row=2, column=0,
                             sticky='w', padx=30, pady=(0,10))

        # Brain cards row
        self.cards_frame = tk.Frame(f, bg=BG_MAIN)
        self.cards_frame.grid(row=3, column=0, sticky='ew',
                              padx=30)
        self.cards_frame.columnconfigure(0, weight=1)
        self.cards_frame.columnconfigure(1, weight=1)
        self.cards_frame.columnconfigure(2, weight=1)

        self.brain_cards = []
        for i in range(3):
            card = tk.Frame(self.cards_frame, bg=BG_CARD2,
                            highlightbackground=BORDER,
                            highlightthickness=1)
            card.grid(row=0, column=i, sticky='ew',
                      padx=(0 if i==0 else 10, 0), pady=0)
            self.brain_cards.append(card)

        # Bottom row — summary & recommendation
        bottom = tk.Frame(f, bg=BG_MAIN)
        bottom.grid(row=4, column=0, sticky='ew',
                    padx=30, pady=15)
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)

        # Summary card
        sum_card = tk.Frame(bottom, bg=BG_CARD,
                            highlightbackground=BORDER,
                            highlightthickness=1)
        sum_card.grid(row=0, column=0, sticky='nsew',
                      padx=(0,10))

        tk.Label(sum_card, text='📄  SUMMARY',
                 bg=BG_CARD, fg=TEXT_SEC,
                 font=('Helvetica', 9, 'bold')).pack(
            anchor='w', padx=16, pady=(12,4))

        self.summary_text = tk.Label(sum_card,
            text='Run prediction to see summary.',
            bg=BG_CARD, fg=TEXT_SEC,
            font=('Helvetica', 10),
            wraplength=280, justify='left')
        self.summary_text.pack(anchor='w',
                               padx=16, pady=(0,14))

        # Recommendation card
        rec_card = tk.Frame(bottom, bg=BG_CARD,
                            highlightbackground=BORDER,
                            highlightthickness=1)
        rec_card.grid(row=0, column=1, sticky='nsew')

        tk.Label(rec_card, text='🩺  RECOMMENDATION',
                 bg=BG_CARD, fg=TEXT_SEC,
                 font=('Helvetica', 9, 'bold')).pack(
            anchor='w', padx=16, pady=(12,4))

        self.rec_text = tk.Label(rec_card,
            text='Run prediction to see recommendation.',
            bg=BG_CARD, fg=TEXT_SEC,
            font=('Helvetica', 10),
            wraplength=280, justify='left')
        self.rec_text.pack(anchor='w', padx=16, pady=(0,14))

    def update_result_page(self, nn_type, nn_proba,
                            rf_type):
        color = STROKE_COLORS.get(nn_type, TEXT_PRI)
        conf  = max(nn_proba) * 100

        # Main diagnosis
        self.diag_icon.config(text='✔' if nn_type != 'No Stroke'
                              else '✓', fg=color)
        self.diag_sub.config(fg=color)
        self.diag_label.config(
            text=f'{nn_type.upper()} STROKE'
                  if nn_type != 'No Stroke'
                  else 'NO STROKE DETECTED',
            fg=color)
        self.conf_label.config(
            text=f'Confidence Score\n{conf:.2f}%',
            fg=color,
            font=('Helvetica', 13, 'bold'))

        # Brain cards — show 3 stroke types
        display_types = ['Ischemic', 'Hemorrhagic', 'TIA']
        display_probs = [nn_proba[1], nn_proba[2], nn_proba[3]]
        disp_colors   = [GREEN, RED, ORANGE]
        borders       = ['#1a1a1a','#1a1a1a','#1a1a1a']

        # Highlight predicted card
        for i, t in enumerate(display_types):
            if t == nn_type:
                borders[i] = disp_colors[i]

        for i, card in enumerate(self.brain_cards):
            for w in card.winfo_children():
                w.destroy()

            stype = display_types[i]
            prob  = display_probs[i]
            col   = disp_colors[i]
            bord  = borders[i]

            card.config(highlightbackground=bord,
                        highlightthickness=2 if bord != '#1a1a1a' else 1)

            tk.Label(card, text=f'{stype} Stroke',
                     bg=BG_CARD2, fg=col,
                     font=('Helvetica', 11, 'bold')).pack(
                pady=(12,6))

            # Brain visualization
            brain = BrainImageCard(card, stype, prob, col)
            brain.pack(pady=4)
            # Percentage
            tk.Label(card, text=f'{prob*100:.2f}%',
                     bg=BG_CARD2, fg=col,
                     font=('Helvetica', 14, 'bold')).pack()

            # Bar
            bar_frame = tk.Frame(card, bg=BG_CARD2)
            bar_frame.pack(fill='x', padx=16, pady=(4,16))

            track = tk.Frame(bar_frame, bg='#1a1a2e', height=6)
            track.pack(fill='x')
            track.pack_propagate(False)

            fill_w = max(2, int(prob * 200))
            fill   = tk.Frame(track, bg=col,
                              width=fill_w, height=6)
            fill.place(x=0, y=0)

        # Summary
        if nn_type != 'No Stroke':
            self.summary_text.config(
                text=f'The model predicts the patient most likely '
                     f'has an {nn_type} Stroke with a confidence '
                     f'of {conf:.2f}%.',
                fg=TEXT_PRI)
            self.rec_text.config(
                text='Please consult a neurologist for further '
                     'evaluation and treatment immediately.',
                fg=RED)
        else:
            self.summary_text.config(
                text=f'The model predicts No Stroke with a '
                     f'confidence of {conf:.2f}%. Patient appears '
                     f'to be at low risk.',
                fg=TEXT_PRI)
            self.rec_text.config(
                text='Continue regular health monitoring. '
                     'Maintain healthy lifestyle habits.',
                fg=GREEN)

    # ── About Page ───────────────────────────────────────────
    def build_about_page(self):
        f = self.about_frame
        f.columnconfigure(0, weight=1)

        tk.Label(f, text='About This System',
                 bg=BG_MAIN, fg=TEXT_PRI,
                 font=('Helvetica', 16, 'bold')).grid(
            row=0, column=0, sticky='w',
            padx=30, pady=(20,15))

        items = [
            ('🧠 System',
             'ML-Based Brain Stroke Diagnosis & Prediction System'),
            ('📊 Models',
             'Logistic Regression, Random Forest, KNN, '
             'Neural Network (Deep Learning)'),
            ('🎯 Purpose',
             'Classify three types of brain stroke: '
             'Ischemic, Hemorrhagic, and TIA'),
            ('📚 Dataset',
             'Brain Stroke Dataset with SMOTE balancing'),
            ('⚠️ Disclaimer',
             'For educational and research purposes only. '
             'Not a medical diagnosis tool.'),
        ]

        for i, (title, desc) in enumerate(items):
            card = tk.Frame(f, bg=BG_CARD,
                            highlightbackground=BORDER,
                            highlightthickness=1)
            card.grid(row=i+1, column=0, sticky='ew',
                      padx=30, pady=5)
            tk.Label(card, text=title, bg=BG_CARD,
                     fg=CYAN,
                     font=('Helvetica', 10, 'bold')).pack(
                anchor='w', padx=16, pady=(10,2))
            tk.Label(card, text=desc, bg=BG_CARD,
                     fg=TEXT_SEC,
                     font=('Helvetica', 10),
                     wraplength=700,
                     justify='left').pack(
                anchor='w', padx=16, pady=(0,10))

    # ── Encode ───────────────────────────────────────────────
    def encode_inputs(self):
        gender   = 1 if self.gender_var.get()=='Male' else 0
        age      = float(self.age_var.get())
        hyp      = 1 if self.hyp_var.get()=='Yes' else 0
        hd       = 1 if self.hd_var.get()=='Yes' else 0
        married  = 1 if self.married_var.get()=='Yes' else 0
        work_map = {'Private':2,'Govt':0,
                    'Self-employed':3,'Children':1,
                    'Never worked':4}
        work     = work_map[self.work_var.get()]
        res      = 1 if self.res_var.get()=='Urban' else 0
        glucose  = float(self.glucose_var.get())
        bmi      = float(self.bmi_var.get())
        smoke_map= {'never smoked':2,'formerly smoked':1,
                    'smokes':3,'Unknown':0}
        smoke    = smoke_map[self.smoke_var.get()]
        return np.array([[gender,age,hyp,hd,married,
                          work,res,glucose,bmi,smoke]])

    # ── Predict ──────────────────────────────────────────────
    def predict(self):
        if not models_loaded:
            messagebox.showerror("Error",
                "Run stroke_advanced.py first!")
            return
        try:
            patient = self.encode_inputs()
            scaled  = scaler.transform(patient)

            nn_proba = nn_model.predict(scaled, verbose=0)[0]
            nn_pred  = int(np.argmax(nn_proba))
            rf_pred  = int(rf_model.predict(scaled)[0])

            nn_type = stroke_labels[nn_pred]
            rf_type = stroke_labels[rf_pred]

            self.update_result_page(nn_type, nn_proba, rf_type)
            self.show_page('result')

        except ValueError:
            messagebox.showerror("Input Error",
                "Please enter valid numbers for Age, "
                "Glucose, and BMI.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Clear ────────────────────────────────────────────────
    def clear_form(self):
        self.gender_var.set('Male')
        self.age_var.set('67')
        self.hyp_var.set('Yes')
        self.hd_var.set('Yes')
        self.married_var.set('Yes')
        self.work_var.set('Private')
        self.res_var.set('Urban')
        self.glucose_var.set('185.0')
        self.bmi_var.set('34.5')
        self.smoke_var.set('formerly smoked')

# ── Run ──────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    app  = StrokeApp(root)
    root.mainloop()