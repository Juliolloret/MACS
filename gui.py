import sys
import os
import threading
import time  # Added for small delay in closeEvent
import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QTextEdit, QMessageBox,
    QGroupBox, QSizePolicy, QFrame
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

# --- Import the backend logic ---
backend_imported_successfully = False
backend_import_error_message = ""
try:
    # This assumes multi_agent_llm_system.py is in the same directory or PYTHONPATH
    import multi_agent_llm_system as backend

    backend_imported_successfully = True
except ImportError as e:
    backend_import_error_message = f"Initial import failed: {e}\n"
    # Attempt to add the script's directory to sys.path for local imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    try:
        import multi_agent_llm_system as backend

        backend_imported_successfully = True
        backend_import_error_message = ""
    except ImportError as e2:
        backend_import_error_message += f"Second import attempt failed: {e2}\nCould not import backend script 'multi_agent_llm_system.py'. Ensure it's in the same directory or PYTHONPATH.\nCurrent sys.path: {sys.path}"


# --- Worker Signals for Threading ---
class WorkerSignals(QObject):
    finished_all = pyqtSignal()
    error = pyqtSignal(tuple)  # (exception_type, exception_value, traceback_str, context_info)
    progress = pyqtSignal(str)
    pdf_processed = pyqtSignal(
        str)  # May be less relevant for fully integrated mode, but can signal individual PDF loading/summarizing steps


class AgentAppGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Agent Research Assistant - Integrated Analysis")
        self.setGeometry(100, 100, 950, 850)

        self.processing_thread = None
        self.signals = WorkerSignals()
        self.stop_event = threading.Event()

        self.signals.progress.connect(self.log_status_to_gui)
        self.signals.finished_all.connect(self.on_all_workflows_finished)
        self.signals.error.connect(self.on_workflow_error)
        self.signals.pdf_processed.connect(self.on_single_pdf_processed)

        self.backend_ok = backend_imported_successfully
        if not self.backend_ok:
            QMessageBox.critical(self, "Critical Backend Import Error",
                                 backend_import_error_message + "\nThe application will have limited functionality.")

        self.init_ui()

        if self.backend_ok:
            try:
                # Backend's load_app_config resolves "config.json" relative to its SCRIPT_DIR
                if backend.load_app_config():
                    backend.set_status_callback(self.signals.progress.emit)
                    self.log_status_to_gui("[GUI] Backend default configuration (config.json) loaded successfully.")
                else:
                    self.log_status_to_gui(
                        "[GUI] ERROR: Backend default configuration (config.json) failed to load. Check backend logs for details.")
            except Exception as e:
                self.log_status_to_gui(f"[GUI] ERROR loading backend default configuration: {e}")
                QMessageBox.critical(self, "Config Load Error",
                                     f"An unexpected error occurred while loading the default backend configuration (config.json).\nError: {e}")
                self.backend_ok = False  # Treat as backend not okay
                if hasattr(self, 'start_button'):  # Ensure button exists
                    self.start_button.setEnabled(False)
                    self.start_button.setText("Backend Config Error")
        else:
            self.log_status_to_gui(
                "[GUI] CRITICAL: Backend module could not be imported. Functionality will be severely limited.")
            if hasattr(self, 'start_button'):
                self.start_button.setEnabled(False)
                self.start_button.setText("Backend Import Error")

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

        setup_group.setLayout(setup_layout)
        self.main_layout.addWidget(setup_group)

        start_button_layout = QHBoxLayout()
        start_button_layout.addStretch()
        self.start_button = QPushButton("Start Integrated Analysis")
        self.start_button.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.start_button.setMinimumHeight(35)
        self.start_button.clicked.connect(self.start_integrated_workflow_thread)
        start_button_layout.addWidget(self.start_button)
        start_button_layout.addStretch()
        self.main_layout.addLayout(start_button_layout)
        self.main_layout.addSpacing(10)

        monitor_group = QGroupBox("Workflow Monitor")
        monitor_layout = QVBoxLayout()
        monitor_layout.setContentsMargins(10, 10, 10, 10)
        self.log_text_area = QTextEdit()
        self.log_text_area.setReadOnly(True)
        monitor_layout.addWidget(self.log_text_area)
        monitor_group.setLayout(monitor_layout)
        self.main_layout.addWidget(monitor_group, 1)

        if not self.backend_ok:
            self.start_button.setEnabled(False)
            self.start_button.setText("Backend Error - Cannot Start")

    def log_status_to_gui(self, message):
        self.log_text_area.append(message)
        QTimer.singleShot(0, lambda: self.log_text_area.verticalScrollBar().setValue(
            self.log_text_area.verticalScrollBar().maximum()))

    def browse_config_file(self):
        current_path = self.config_file_entry.text()
        start_dir = os.path.dirname(current_path) if current_path and os.path.exists(
            os.path.dirname(current_path)) else os.path.expanduser("~")
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Configuration File", start_dir, "JSON files (*.json)")
        if filepath:
            self.config_file_entry.setText(filepath)

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

        self.processing_thread = threading.Thread(
            target=self.run_integrated_backend_task,
            args=(
            all_pdf_files_in_folder, exp_data_file_path, project_specific_output_dir, config_file_path_for_backend)
        )
        self.processing_thread.start()

    def run_integrated_backend_task(self, pdf_file_paths_list, exp_data_path, project_output_dir, config_file_path):
        try:
            if self.stop_event.is_set():
                self.signals.progress.emit("[GUI] Integrated processing cancelled before start.")
                return

            self.signals.progress.emit(f"\n[GUI] Starting integrated analysis for the project...")

            # This is the corrected call
            result = backend.run_project_orchestration(
                pdf_file_paths=pdf_file_paths_list,
                experimental_data_path=exp_data_path,
                project_base_output_dir=project_output_dir,
                status_update_callback=self.signals.progress.emit,
                config_file_path=config_file_path
            )

            if self.stop_event.is_set():
                self.signals.progress.emit(f"[GUI] Integrated project processing was interrupted.")
            elif result and result.get("error"):
                self.signals.error.emit(("ProjectError", result.get("error"), "See logs for details", "Project Level"))
            else:
                self.signals.progress.emit(f"[GUI] Integrated project analysis finished successfully.")

        except Exception as e:
            if self.stop_event.is_set():
                self.signals.progress.emit(f"[GUI] Exception during integrated processing while stopping: {e}")
            else:
                import traceback
                self.signals.error.emit((type(e), e, traceback.format_exc(), "Integrated Project Level"))
        finally:
            self.signals.finished_all.emit()

    def on_single_pdf_processed(self, pdf_filename):
        # This can be used to log initial loading/summarizing if the backend emits such signals
        # For now, it's less critical in the fully integrated flow.
        # self.log_status_to_gui(f"[GUI] Initial processing for {pdf_filename} completed (e.g., summary generated).")
        pass

    def on_all_workflows_finished(self):
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

    def on_workflow_error(self, error_tuple):
        exctype, value, tb_str, context_info = error_tuple
        self.log_status_to_gui(f"[GUI] ERROR during '{context_info}': {value}")
        self.log_status_to_gui(f"[GUI] Traceback for '{context_info}':\n{tb_str}")

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.is_alive():
            reply = QMessageBox.question(self, 'Confirm Quit',
                                         "An analysis is currently processing. Are you sure you want to quit? Attempting to stop gracefully.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.log_status_to_gui("[GUI] Attempting to stop worker thread...")
                self.stop_event.set()

                QApplication.processEvents()
                time.sleep(0.1)

                if self.processing_thread.is_alive():
                    self.log_status_to_gui(f"[GUI] Waiting for worker thread to join (max 5s)...")
                    self.processing_thread.join(timeout=5.0)

                if self.processing_thread.is_alive():
                    self.log_status_to_gui(
                        "[GUI] Worker thread did not stop in time. The application will now close. The Python process might linger if the thread is stuck in a non-interruptible operation.")
                else:
                    self.log_status_to_gui("[GUI] Worker thread stopped or finished.")

                event.accept()
                QApplication.instance().quit()
            else:
                event.ignore()
        else:
            event.accept()
            QApplication.instance().quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    if not backend_imported_successfully:
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setText("Critical Backend Import Error")
        error_box.setInformativeText(backend_import_error_message + "\nThe application cannot start correctly.")
        error_box.setWindowTitle("Application Startup Error")
        error_box.exec()
        sys.exit(1)

    main_window = AgentAppGUI()
    main_window.show()
    sys.exit(app.exec())
