"""Graphical user interface for the Multi-Agent Research Assistant."""

import datetime
import os
import subprocess
import sys
import threading
import time  # Added for small delay in closeEvent
import traceback
import configparser

# pylint: disable=missing-function-docstring, import-error, broad-exception-caught


def install_dependencies():
    """Install dependencies from ``requirements.txt`` if present."""
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if not os.path.exists(requirements_path):
        print("[Dependency Check] requirements.txt not found. Skipping installation.")
        return

    print("[Dependency Check] Checking and installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("[Dependency Check] All dependencies are satisfied.")
    except subprocess.CalledProcessError:
        print(
            "[Dependency Check] Failed to install dependencies. Please install them manually using 'pip install -r requirements.txt'"
        )
    except FileNotFoundError:
        print("[Dependency Check] 'pip' command not found. Please ensure pip is installed and in your PATH.")

try:
    from PyQt6.QtCore import QTimer, QObject, pyqtSignal, Qt
    from PyQt6.QtGui import QAction, QFont, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMenuBar,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    # PyQt6 enums
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    KEEP_ASPECT_RATIO = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH_TRANSFORMATION = Qt.TransformationMode.SmoothTransformation
    MB_YES = QMessageBox.StandardButton.Yes
    MB_NO = QMessageBox.StandardButton.No
    MB_CRITICAL = QMessageBox.Icon.Critical
    FONT_BOLD = QFont.Weight.Bold
except ImportError:  # pragma: no cover
    try:
        from PyQt5.QtCore import QTimer, QObject, pyqtSignal, Qt
        from PyQt5.QtGui import QAction, QFont, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QFileDialog,
            QDialog,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMenuBar,
            QMessageBox,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        # PyQt5 enums
        ALIGN_CENTER = Qt.AlignCenter
        KEEP_ASPECT_RATIO = Qt.KeepAspectRatio
        SMOOTH_TRANSFORMATION = Qt.SmoothTransformation
        MB_YES = QMessageBox.Yes
        MB_NO = QMessageBox.No
        MB_CRITICAL = QMessageBox.Critical
        try:
            FONT_BOLD = QFont.Weight.Bold  # type: ignore[attr-defined]
        except AttributeError:  # PyQt5 fallback
            FONT_BOLD = QFont.Bold
    except ImportError as import_error:  # pragma: no cover
        raise ImportError(
            "PyQt5 or PyQt6 is required to run the GUI."
        ) from import_error

# --- Import adaptive cycle for single-pass evolution ---
from adaptive.adaptive_graph_runner import adaptive_cycle
# --- Import the backend logic ---
BACKEND_IMPORTED_SUCCESSFULLY = False
BACKEND_IMPORT_ERROR_MESSAGE = ""
try:
    # This assumes multi_agent_llm_system.py is in the same directory or PYTHONPATH
    import multi_agent_llm_system as backend

    BACKEND_IMPORTED_SUCCESSFULLY = True
except ImportError as e:
    BACKEND_IMPORT_ERROR_MESSAGE = f"Initial import failed: {e}\n"
    # Attempt to add the script's directory to sys.path for local imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    try:
        import multi_agent_llm_system as backend

        BACKEND_IMPORTED_SUCCESSFULLY = True
        BACKEND_IMPORT_ERROR_MESSAGE = ""
    except ImportError as e2:
        BACKEND_IMPORT_ERROR_MESSAGE += (
            "Second import attempt failed: "
            f"{e2}\nCould not import backend script 'multi_agent_llm_system.py'. "
            "Ensure it's in the same directory or PYTHONPATH.\nCurrent sys.path: "
            f"{sys.path}"
        )


# --- Worker Signals for Threading ---


class WorkerSignals(QObject):
    """Signals used by background worker threads."""

    finished_all = pyqtSignal()
    error = pyqtSignal(tuple)  # (exception_type, exception_value, traceback_str, context_info)
    progress = pyqtSignal(str)
    pdf_processed = pyqtSignal(
        str
    )  # May be less relevant for fully integrated mode, but can signal individual PDF loading/summarizing steps
    graph_updated = pyqtSignal(str)  # Path to updated graph image


class AgentAppGUI(QWidget):
    """Main application window for the Research Assistant GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Agent Research Assistant - Integrated Analysis")
        self.setGeometry(100, 100, 950, 850)

        self.processing_thread = None
        self.signals = WorkerSignals()
        self.stop_event = threading.Event()
        self.active_task = None

        self.signals.progress.connect(self.log_status_to_gui)
        self.signals.finished_all.connect(self.on_all_workflows_finished)
        self.signals.error.connect(self.on_workflow_error)
        self.signals.pdf_processed.connect(self.on_single_pdf_processed)
        self.signals.graph_updated.connect(self.update_graph_display)

        self.backend_ok = BACKEND_IMPORTED_SUCCESSFULLY
        if not self.backend_ok:
            QMessageBox.critical(
                self,
                "Critical Backend Import Error",
                BACKEND_IMPORT_ERROR_MESSAGE + "\nThe application will have limited functionality.",
            )

        self.init_ui()
        self.create_menus()
        self.routes_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predefined_routes.ini")
        self.predefined_routes = {}
        if os.path.exists(self.routes_file):
            self.load_predefined_routes(self.routes_file, show_message=False)

        if self.backend_ok:
            # The config is now loaded on-demand when the workflow starts.
            # We can still set the status callback here.
            backend.set_status_callback(self.signals.progress.emit)
            backend.set_graph_callback(self.signals.graph_updated.emit)
            self.log_status_to_gui("[GUI] Backend module loaded. Configuration will be loaded on workflow start.")
        else:
            self.log_status_to_gui(
                "[GUI] CRITICAL: Backend module could not be imported. Functionality will be severely limited.")
            if hasattr(self, 'start_button'):
                self.start_button.setEnabled(False)
                self.start_button.setText("Backend Import Error")
            if hasattr(self, 'evolution_button'):
                self.evolution_button.setEnabled(False)
                self.evolution_button.setText("Backend Import Error")

        self.apply_stylesheet("light")

    def apply_stylesheet(self, theme="light"):
        # Stylesheet from previous version (gui_py_v3_shutdown_fix)
        base_style = """
            QWidget { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 10pt; 
            }
            QGroupBox { 
                font-weight: bold; 
                font-size: 11pt; 
                margin-top: 15px; 
                padding-top: 20px; 
                border: 1px solid #d0d0d0;
                border-radius: 5px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left; 
                padding: 0 5px 0 5px;
                margin-left: 10px;
            }
            QLabel { 
                font-weight: normal; 
                padding: 2px;
            }
            QLineEdit, QTextEdit { 
                border: 1px solid #c0c0c0; 
                border-radius: 4px; 
                padding: 5px; 
                background-color: #ffffff;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #0078d4; 
            }
            QTextEdit {
                 font-family: 'Consolas', 'Courier New', monospace;
                 font-size: 9.5pt;
            }
            QPushButton { 
                padding: 8px 15px; 
                border-radius: 4px; 
                font-weight: bold;
                min-height: 20px; 
            }
            QPushButton:disabled { 
                background-color: #e0e0e0; 
                border-color: #c0c0c0; 
                color: #a0a0a0;
            }
        """
        if theme == "light":
            light_theme_style = base_style + """
                QWidget { 
                    background-color: #f0f0f0; 
                    color: #222222; 
                }
                QGroupBox {
                    border-color: #d0d0d0;
                    background-color: #f8f8f8; 
                }
                QGroupBox::title {
                     color: #005a9e; 
                }
                QLineEdit, QTextEdit {
                    border-color: #cccccc;
                    background-color: #ffffff;
                    color: #333333;
                }
                QPushButton { 
                    background-color: #0078d4; 
                    color: white; 
                    border: 1px solid #005a9e; 
                }
                QPushButton:hover { 
                    background-color: #005a9e; 
                    border: 1px solid #004578;
                }
                QPushButton:pressed {
                    background-color: #004578; 
                }
            """
            self.setStyleSheet(light_theme_style)

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)

        setup_group = QGroupBox("Project Configuration")
        setup_layout = QVBoxLayout()
        setup_layout.setSpacing(10)
        setup_layout.setContentsMargins(10, 10, 10, 10)

        pn_layout = QHBoxLayout()
        pn_layout.addWidget(QLabel("Project Name:"))
        self.project_name_entry = QLineEdit("IntegratedResearchProject")
        pn_layout.addWidget(self.project_name_entry)
        setup_layout.addLayout(pn_layout)

        pdf_folder_layout = QHBoxLayout()
        pdf_folder_layout.addWidget(QLabel("Input PDF Folder(s):"))
        self.pdf_folder_path_entry = QLineEdit()
        self.pdf_folder_path_entry.setPlaceholderText("Select folder containing PDF files...")
        self.pdf_folder_path_entry.setReadOnly(True)
        pdf_folder_layout.addWidget(self.pdf_folder_path_entry)
        self.browse_pdf_folder_button = QPushButton("Browse PDFs")
        self.browse_pdf_folder_button.clicked.connect(self.browse_pdf_folder)
        pdf_folder_layout.addWidget(self.browse_pdf_folder_button)
        setup_layout.addLayout(pdf_folder_layout)

        exp_data_layout = QHBoxLayout()
        exp_data_layout.addWidget(QLabel("Experimental Data File (Optional Text):"))
        self.exp_data_file_entry = QLineEdit()
        self.exp_data_file_entry.setPlaceholderText("Select a text file with experimental data summary...")
        self.exp_data_file_entry.setReadOnly(True)
        exp_data_layout.addWidget(self.exp_data_file_entry)
        self.browse_exp_data_button = QPushButton("Browse Exp. Data")
        self.browse_exp_data_button.clicked.connect(self.browse_exp_data_file)
        exp_data_layout.addWidget(self.browse_exp_data_button)
        setup_layout.addLayout(exp_data_layout)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Project Output Directory:"))
        self.output_dir_entry = QLineEdit()
        self.output_dir_entry.setPlaceholderText("Select directory for the integrated project output...")
        self.output_dir_entry.setReadOnly(True)
        try:
            docs_path = os.path.expanduser("~/Documents")
            if not os.path.isdir(docs_path): docs_path = os.getcwd()
        except Exception:
            docs_path = os.getcwd()
        default_output_path = os.path.join(docs_path, "MultiAgent_IntegratedOutputs")
        self.output_dir_entry.setText(default_output_path)

        output_dir_layout.addWidget(self.output_dir_entry)
        self.browse_output_dir_button = QPushButton("Browse...")
        self.browse_output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.browse_output_dir_button)
        setup_layout.addLayout(output_dir_layout)

        config_file_layout = QHBoxLayout()
        config_file_layout.addWidget(QLabel("Config File (Optional):"))
        self.config_file_entry = QLineEdit()
        self.config_file_entry.setPlaceholderText("Default: config.json (located with backend script)")
        self.config_file_entry.setReadOnly(True)
        config_file_layout.addWidget(self.config_file_entry)
        self.browse_config_button = QPushButton("Browse...")
        self.browse_config_button.clicked.connect(self.browse_config_file)
        config_file_layout.addWidget(self.browse_config_button)
        setup_layout.addLayout(config_file_layout)

        experiment_config_layout = QHBoxLayout()
        experiment_config_layout.addWidget(QLabel("Experiment Config File:"))
        self.experiment_config_entry = QLineEdit()
        self.experiment_config_entry.setPlaceholderText("Select experiment optimization config JSON...")
        self.experiment_config_entry.setReadOnly(True)
        experiment_config_layout.addWidget(self.experiment_config_entry)
        self.browse_experiment_config_button = QPushButton("Browse...")
        self.browse_experiment_config_button.clicked.connect(self.browse_experiment_config_file)
        experiment_config_layout.addWidget(self.browse_experiment_config_button)
        setup_layout.addLayout(experiment_config_layout)

        setup_group.setLayout(setup_layout)
        self.main_layout.addWidget(setup_group)

        start_button_layout = QHBoxLayout()
        start_button_layout.addStretch()
        self.start_button = QPushButton("Start Integrated Analysis")
        self.start_button.setFont(QFont('Segoe UI', 12, FONT_BOLD))
        self.start_button.setMinimumHeight(35)
        self.start_button.clicked.connect(self.start_integrated_workflow_thread)
        start_button_layout.addWidget(self.start_button)
        start_button_layout.addStretch()
        self.main_layout.addLayout(start_button_layout)

        evolution_button_layout = QHBoxLayout()
        evolution_button_layout.addStretch()
        self.evolution_button = QPushButton("Run Single Evolution Pass")
        self.evolution_button.setFont(QFont('Segoe UI', 12, FONT_BOLD))
        self.evolution_button.setMinimumHeight(35)
        self.evolution_button.clicked.connect(self.start_evolution_pass_thread)
        evolution_button_layout.addWidget(self.evolution_button)
        evolution_button_layout.addStretch()
        self.main_layout.addLayout(evolution_button_layout)
        self.main_layout.addSpacing(10)

        monitor_group = QGroupBox("Workflow Monitor")
        monitor_layout = QVBoxLayout()
        monitor_layout.setContentsMargins(10, 10, 10, 10)
        self.log_text_area = QTextEdit()
        self.log_text_area.setReadOnly(True)
        monitor_layout.addWidget(self.log_text_area)
        monitor_group.setLayout(monitor_layout)
        self.main_layout.addWidget(monitor_group, 1)

        graph_group = QGroupBox("Graph Visualization")
        graph_layout = QVBoxLayout()
        self.graph_label = QLabel("Graph will be displayed here once available.")
        self.graph_label.setAlignment(ALIGN_CENTER)
        graph_layout.addWidget(self.graph_label)
        graph_group.setLayout(graph_layout)
        self.main_layout.addWidget(graph_group)

        if not self.backend_ok:
            self.start_button.setEnabled(False)
            self.start_button.setText("Backend Error - Cannot Start")

    def create_menus(self):
        self.menubar = QMenuBar(self)
        file_menu = self.menubar.addMenu("File")

        save_log_action = QAction("Save Log", self)
        save_log_action.triggered.connect(self.save_log)
        file_menu.addAction(save_log_action)

        export_graph_action = QAction("Export Graph as JPG", self)
        export_graph_action.triggered.connect(self.save_graph_as_jpg)
        file_menu.addAction(export_graph_action)

        routes_menu = self.menubar.addMenu("Routes")
        load_routes_action = QAction("Load Predefined Routes", self)
        load_routes_action.triggered.connect(self.load_predefined_routes)
        routes_menu.addAction(load_routes_action)

        graph_menu = self.menubar.addMenu("Graph")
        define_graph_action = QAction("Define Graph", self)
        define_graph_action.triggered.connect(self.open_graph_definition_window)
        graph_menu.addAction(define_graph_action)

        self.main_layout.setMenuBar(self.menubar)

    def save_log(self):
        if not self.log_text_area.toPlainText().strip():
            QMessageBox.information(self, "Save Log", "There is no log to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log",
            os.path.expanduser("~"),
            "Text Files (*.txt)",
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.log_text_area.toPlainText())
                QMessageBox.information(self, "Save Log", f"Log saved to: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Save Log", f"Failed to save log: {e}")

    def save_graph_as_jpg(self):
        pixmap = self.graph_label.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.information(self, "Export Graph", "No graph image available to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Graph",
            os.path.expanduser("~"),
            "JPEG Images (*.jpg *.jpeg)",
        )
        if filepath:
            if not filepath.lower().endswith(('.jpg', '.jpeg')):
                filepath += '.jpg'
            try:
                pixmap.save(filepath, "JPG")
                QMessageBox.information(self, "Export Graph", f"Graph saved to: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Export Graph", f"Failed to save graph: {e}")

    def load_predefined_routes(self, file_path=None, show_message=True):
        if not file_path:
            start_dir = os.path.dirname(self.routes_file) if hasattr(self, 'routes_file') else os.getcwd()
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Routes INI File",
                start_dir,
                "INI files (*.ini)",
            )
            if not file_path:
                return
        config = configparser.ConfigParser()
        try:
            config.read(file_path)
            if config.has_section('routes'):
                self.predefined_routes = dict(config.items('routes'))
            else:
                self.predefined_routes = {}
            self.routes_file = file_path
            if show_message:
                QMessageBox.information(
                    self,
                    "Predefined Routes",
                    f"Loaded {len(self.predefined_routes)} routes from {file_path}",
                )
        except Exception as e:
            if show_message:
                QMessageBox.warning(self, "Predefined Routes", f"Failed to load routes: {e}")

    def open_graph_definition_window(self):
        dialog = GraphDefinitionWindow(self)
        dialog.exec()

    def log_status_to_gui(self, message):
        self.log_text_area.append(message)
        QTimer.singleShot(0, lambda: self.log_text_area.verticalScrollBar().setValue(
            self.log_text_area.verticalScrollBar().maximum()))

    def update_graph_display(self, image_path: str):
        """Update the graph visualization label with the image at ``image_path``."""

        if not image_path:
            return
        image_path = image_path.strip().strip("\"")
        if os.path.exists(image_path) and image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.graph_label.width() or 1,
                    self.graph_label.height() or 1,
                    KEEP_ASPECT_RATIO,
                    SMOOTH_TRANSFORMATION,
                )
                self.graph_label.setPixmap(scaled)
            else:
                self.graph_label.setText(f"Failed to load graph image: {image_path}")
        else:
            self.graph_label.setText(f"Graph saved to: {image_path}")

    def browse_config_file(self):
        current_path = self.config_file_entry.text()
        start_dir = os.path.dirname(current_path) if current_path and os.path.exists(
            os.path.dirname(current_path)) else os.path.expanduser("~")
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Configuration File", start_dir, "JSON files (*.json)")
        if filepath:
            self.config_file_entry.setText(filepath)

    def browse_experiment_config_file(self):
        """Open a dialog to choose an experiment configuration file."""
        current_path = self.experiment_config_entry.text()
        start_dir = (
            os.path.dirname(current_path)
            if current_path and os.path.exists(os.path.dirname(current_path))
            else os.path.expanduser("~")
        )
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Experiment Config File", start_dir, "JSON files (*.json)"
        )
        if filepath:
            self.experiment_config_entry.setText(filepath)

    def browse_pdf_folder(self):
        current_path = self.pdf_folder_path_entry.text()
        start_dir = current_path if current_path and os.path.isdir(current_path) else os.path.expanduser("~")
        dirpath = QFileDialog.getExistingDirectory(self, "Select Folder Containing Input PDFs", start_dir)
        if dirpath:
            self.pdf_folder_path_entry.setText(dirpath)

    def browse_exp_data_file(self):
        current_path = self.exp_data_file_entry.text()
        start_dir = os.path.dirname(current_path) if current_path and os.path.exists(
            os.path.dirname(current_path)) else os.path.expanduser("~")
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Experimental Data File (Text)", start_dir,
                                                  "Text files (*.txt);;All files (*)")
        if filepath:
            self.exp_data_file_entry.setText(filepath)

    def browse_output_dir(self):
        current_path = self.output_dir_entry.text()
        start_dir = current_path if current_path and os.path.isdir(current_path) else os.path.expanduser("~")
        dirpath = QFileDialog.getExistingDirectory(self, "Select Directory for Project Output", start_dir)
        if dirpath:
            self.output_dir_entry.setText(dirpath)
        elif not self.output_dir_entry.text():
            try:
                docs_path = os.path.expanduser("~/Documents")
                if not os.path.isdir(docs_path): docs_path = os.getcwd()
            except Exception:
                docs_path = os.getcwd()
            self.output_dir_entry.setText(os.path.join(docs_path, "MultiAgent_IntegratedOutputs"))

    def start_integrated_workflow_thread(self):
        if not self.backend_ok:
            QMessageBox.critical(self, "Backend Error",
                                 "Backend module not loaded or config error. Cannot start workflow.")
            return

        pdf_folder_path = self.pdf_folder_path_entry.text()
        project_output_base_dir = self.output_dir_entry.text()
        project_name_input = self.project_name_entry.text().strip()
        exp_data_file_path = self.exp_data_file_entry.text().strip()

        config_file_path_gui = self.config_file_entry.text().strip()
        config_file_path_for_backend = config_file_path_gui if config_file_path_gui else "config.json"

        if config_file_path_gui and not os.path.exists(config_file_path_gui):
            QMessageBox.warning(self, "Config File Error",
                                f"Selected configuration file not found:\n{config_file_path_gui}")
            return
        log_msg = f"[GUI] Using configuration: {config_file_path_for_backend}" + (
            " (default)" if not config_file_path_gui else "")
        self.log_status_to_gui(log_msg)

        if not project_name_input:
            QMessageBox.warning(self, "Input Error", "Please enter a Project Name.")
            return
        if not pdf_folder_path or not os.path.isdir(pdf_folder_path):
            QMessageBox.warning(self, "Input Error", "Please select a valid Input PDF Folder.")
            return
        if not project_output_base_dir:
            QMessageBox.warning(self, "Input Error", "Please select a Project Output Base Directory.")
            return

        if exp_data_file_path and not os.path.isfile(exp_data_file_path):
            QMessageBox.warning(self, "Input Error",
                                f"Experimental data file not found: {exp_data_file_path}. Proceeding without it.")
            exp_data_file_path = ""  # Process without it if not found

        all_pdf_files_in_folder = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if
                                   f.lower().endswith(".pdf")]
        if not all_pdf_files_in_folder:
            QMessageBox.information(self, "No PDFs Found", f"No PDF files found in folder:\n{pdf_folder_path}")
            return

        safe_project_name_base = "".join(
            c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name_input).rstrip().replace(" ", "_")
        if not safe_project_name_base: safe_project_name_base = "Unnamed_IntegratedProject"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_specific_output_dir = os.path.join(project_output_base_dir, f"{safe_project_name_base}_{timestamp}")

        try:
            if not os.path.exists(project_specific_output_dir):  # Ensure the specific project output dir is made
                os.makedirs(project_specific_output_dir, exist_ok=True)
            self.log_status_to_gui(f"[GUI] Integrated project output directory set to: {project_specific_output_dir}")
        except OSError as e:
            QMessageBox.critical(self, "Directory Error",
                                 f"Could not create project output directory:\n{project_specific_output_dir}\nError: {e}")
            return

        self.log_text_area.clear()
        self.log_status_to_gui(
            f"[GUI] Starting INTEGRATED workflow for Project: '{project_specific_output_dir}'")  # Log specific dir
        self.log_status_to_gui(f"[GUI] Processing {len(all_pdf_files_in_folder)} PDF(s) from: {pdf_folder_path}")
        if exp_data_file_path:
            self.log_status_to_gui(f"[GUI] Including experimental data from: {exp_data_file_path}")
        else:
            self.log_status_to_gui("[GUI] No experimental data file provided.")

        self.start_button.setText("Processing Project...")
        self.start_button.setEnabled(False)

        self.stop_event.clear()
        self.active_task = "integrated"

        app_config = backend.load_app_config(config_path=config_file_path_for_backend)
        if not app_config:
            QMessageBox.critical(self, "Config Load Error", f"Failed to load configuration from:\n{config_file_path_for_backend}\nCheck logs for details.")
            return

        self.processing_thread = threading.Thread(
            target=self.run_integrated_backend_task,
            args=(
            all_pdf_files_in_folder, exp_data_file_path, project_specific_output_dir, app_config)
        )
        self.processing_thread.start()

    def start_evolution_pass_thread(self):
        """Validate inputs and launch the evolution pass in a background thread."""
        if not self.backend_ok:
            QMessageBox.critical(
                self, "Backend Error", "Backend module not loaded or config error. Cannot start workflow."
            )
            return

        experiment_config_path = self.experiment_config_entry.text().strip()
        if not experiment_config_path or not os.path.exists(experiment_config_path):
            QMessageBox.warning(
                self, "Config File Error", "Please select a valid experiment configuration file."
            )
            return

        pdf_folder_path = self.pdf_folder_path_entry.text()
        project_output_base_dir = self.output_dir_entry.text()
        project_name_input = self.project_name_entry.text().strip()
        exp_data_file_path = self.exp_data_file_entry.text().strip()

        if not project_name_input:
            QMessageBox.warning(self, "Input Error", "Please enter a Project Name.")
            return
        if not pdf_folder_path or not os.path.isdir(pdf_folder_path):
            QMessageBox.warning(self, "Input Error", "Please select a valid Input PDF Folder.")
            return
        if not project_output_base_dir:
            QMessageBox.warning(self, "Input Error", "Please select a Project Output Base Directory.")
            return

        if exp_data_file_path and not os.path.isfile(exp_data_file_path):
            QMessageBox.warning(self, "Input Error",
                                f"Experimental data file not found: {exp_data_file_path}. Proceeding without it.")
            exp_data_file_path = ""

        all_pdf_files_in_folder = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if
                                   f.lower().endswith(".pdf")]
        if not all_pdf_files_in_folder:
            QMessageBox.information(self, "No PDFs Found", f"No PDF files found in folder:\n{pdf_folder_path}")
            return

        safe_project_name_base = "".join(
            c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name_input).rstrip().replace(" ", "_")
        if not safe_project_name_base:
            safe_project_name_base = "Unnamed_Experiment"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_specific_output_dir = os.path.join(project_output_base_dir, f"{safe_project_name_base}_{timestamp}")

        try:
            if not os.path.exists(project_specific_output_dir):
                os.makedirs(project_specific_output_dir, exist_ok=True)
            self.log_status_to_gui(f"[GUI] Experiment output directory set to: {project_specific_output_dir}")
        except OSError as e:
            QMessageBox.critical(self, "Directory Error",
                                 f"Could not create project output directory:\n{project_specific_output_dir}\nError: {e}")
            return

        self.log_text_area.clear()
        self.log_status_to_gui(
            f"[GUI] Starting single evolution pass for Project: '{project_specific_output_dir}'")
        self.log_status_to_gui(f"[GUI] Processing {len(all_pdf_files_in_folder)} PDF(s) from: {pdf_folder_path}")
        if exp_data_file_path:
            self.log_status_to_gui(f"[GUI] Including experimental data from: {exp_data_file_path}")
        else:
            self.log_status_to_gui("[GUI] No experimental data file provided.")

        self.evolution_button.setText("Processing Evolution Pass...")
        self.evolution_button.setEnabled(False)

        self.stop_event.clear()
        self.active_task = "evolution"

        inputs = {
            "initial_inputs": {
                "all_pdf_paths": all_pdf_files_in_folder,
                "experimental_data_file_path": exp_data_file_path,
            },
            "project_base_output_dir": project_specific_output_dir,
        }

        self.processing_thread = threading.Thread(
            target=self.run_evolution_pass_task,
            args=(experiment_config_path, inputs),
        )
        self.processing_thread.start()

    def run_integrated_backend_task(self, pdf_file_paths_list, exp_data_path, project_output_dir, app_config):
        try:
            if self.stop_event.is_set():
                self.signals.progress.emit("[GUI] Integrated processing cancelled before start.")
                return

            self.signals.progress.emit("\n[GUI] Starting integrated analysis for the project...")

            result = backend.run_project_orchestration(
                pdf_file_paths=pdf_file_paths_list,
                experimental_data_path=exp_data_path,
                project_base_output_dir=project_output_dir,
                status_update_callback=self.signals.progress.emit,
                app_config=app_config
            )

            if self.stop_event.is_set():
                self.signals.progress.emit("[GUI] Integrated project processing was interrupted.")
            elif result and result.get("error"):
                self.signals.error.emit(("ProjectError", result.get("error"), "See logs for details", "Project Level"))
            else:
                self.signals.progress.emit("[GUI] Integrated project analysis finished successfully.")

        except Exception as e:
            if self.stop_event.is_set():
                self.signals.progress.emit(f"[GUI] Exception during integrated processing while stopping: {e}")
            else:
                import traceback
                self.signals.error.emit((type(e), e, traceback.format_exc(), "Integrated Project Level"))
        finally:
            self.signals.finished_all.emit()

    def run_evolution_pass_task(self, config_path, inputs):
        """Execute ``adaptive_cycle`` once for the provided experiment configuration."""
        try:
            if self.stop_event.is_set():
                self.signals.progress.emit("[GUI] Evolution pass cancelled before start.")
                return

            self.signals.progress.emit("\n[GUI] Starting single evolution pass...")
            adaptive_cycle(
                config_path=config_path, inputs=inputs, threshold=1.0, max_steps=1
            )
            if self.stop_event.is_set():
                self.signals.progress.emit("[GUI] Evolution pass was interrupted.")
            else:
                self.signals.progress.emit("[GUI] Evolution pass finished successfully.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            if self.stop_event.is_set():
                self.signals.progress.emit(
                    f"[GUI] Exception during evolution pass while stopping: {e}"
                )
            else:
                self.signals.error.emit((type(e), e, traceback.format_exc(), "Evolution Pass"))
        finally:
            self.signals.finished_all.emit()

    def on_single_pdf_processed(self, pdf_filename):
        # This can be used to log initial loading/summarizing if the backend emits such signals
        # For now, it's less critical in the fully integrated flow.
        # self.log_status_to_gui(f"[GUI] Initial processing for {pdf_filename} completed (e.g., summary generated).")
        pass

    def on_all_workflows_finished(self):
        if self.active_task == "evolution":
            self.evolution_button.setText("Run Single Evolution Pass")
            self.evolution_button.setEnabled(True)
            if self.stop_event.is_set():
                self.log_status_to_gui("[GUI] === EVOLUTION PASS STOPPED ===")
                QMessageBox.information(self, "Evolution Pass Stopped",
                                        "The evolution pass was stopped by the user.")
            else:
                self.log_status_to_gui("[GUI] === EVOLUTION PASS FINISHED ===")
                QMessageBox.information(self, "Evolution Pass Complete",
                                        "The single evolution pass has finished.")
        else:
            self.start_button.setText("Start Integrated Analysis")
            self.start_button.setEnabled(True)
            if self.stop_event.is_set():
                self.log_status_to_gui("[GUI] === INTEGRATED WORKFLOW STOPPED ===")
                QMessageBox.information(self, "Integrated Workflow Stopped",
                                        "The integrated analysis process was stopped by the user.")
            else:
                last_log_lines = self.log_text_area.toPlainText().splitlines()
                project_error_logged = any("Project Level" in line and "ERROR" in line for line in last_log_lines[-5:])

                if project_error_logged:
                    self.log_status_to_gui("[GUI] === INTEGRATED WORKFLOW FINISHED WITH ERRORS ===")
                    QMessageBox.warning(self, "Integrated Workflow Finished with Errors",
                                        "The integrated analysis process finished, but errors occurred. Please check the logs.")
                else:
                    self.log_status_to_gui("[GUI] === INTEGRATED WORKFLOW FINISHED SUCCESSFULLY ===")
                    QMessageBox.information(self, "Integrated Workflow Complete",
                                            "The integrated analysis has finished. Check logs and output folder for details.")

        self.processing_thread = None
        self.active_task = None

    def on_workflow_error(self, error_tuple):
        _exc_type, value, tb_str, context_info = error_tuple
        self.log_status_to_gui(f"[GUI] ERROR during '{context_info}': {value}")
        self.log_status_to_gui(f"[GUI] Traceback for '{context_info}':\n{tb_str}")

    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Handle window close events, attempting to stop worker threads."""
        if self.processing_thread and self.processing_thread.is_alive():
            reply = QMessageBox.question(
                self,
                'Confirm Quit',
                "An analysis is currently processing. Are you sure you want to quit? Attempting to stop gracefully.",
                MB_YES | MB_NO,
                MB_NO,
            )
            if reply == MB_YES:
                self.log_status_to_gui("[GUI] Attempting to stop worker thread...")
                self.stop_event.set()

                QApplication.processEvents()
                time.sleep(0.1)

                if self.processing_thread.is_alive():
                    self.log_status_to_gui("[GUI] Waiting for worker thread to join (max 5s)...")
                    self.processing_thread.join(timeout=5.0)

                if self.processing_thread.is_alive():
                    self.log_status_to_gui(
                        "[GUI] Worker thread did not stop in time. The application will now close. The Python process might linger if the thread is stuck in a non-interruptible operation.")
                else:
                    self.log_status_to_gui("[GUI] Worker thread stopped or finished.")

                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class GraphDefinitionWindow(QDialog):
    """Placeholder window for future graph definition features."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Graph Definition")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Graph definition GUI will be implemented in a future iteration."))
        self.setLayout(layout)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run the Multi-Agent Research Assistant GUI.")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies from requirements.txt before launching the GUI.",
    )
    cli_args = parser.parse_args()

    if cli_args.install_deps:
        install_dependencies()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    if not BACKEND_IMPORTED_SUCCESSFULLY:
        error_box = QMessageBox()
        error_box.setIcon(MB_CRITICAL)
        error_box.setText("Critical Backend Import Error")
        error_box.setInformativeText(
            BACKEND_IMPORT_ERROR_MESSAGE + "\nThe application cannot start correctly."
        )
        error_box.setWindowTitle("Application Startup Error")
        error_box.exec()
        sys.exit(1)

    main_window = AgentAppGUI()
    main_window.show()
    sys.exit(app.exec())
