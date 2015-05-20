#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent) : QWidget(parent), _sinogramDisplayActive(false), _crossSectionDisplayActive(false), _reconstructionActive(false), _runAll(false) {
		setAcceptDrops(true);

		_volume.setEmitSignals(true);
		qRegisterMetaType<CtVolume::CompletionStatus>("CtVolume::CompletionStatus");
		QObject::connect(&_volume, SIGNAL(loadingProgress(double)), this, SLOT(reactToLoadProgressUpdate(double)));
		QObject::connect(&_volume, SIGNAL(loadingFinished(CtVolume::CompletionStatus)), this, SLOT(reactToLoadCompletion(CtVolume::CompletionStatus)));
		qRegisterMetaType<cv::Mat>("cv::Mat");
		QObject::connect(&_volume, SIGNAL(reconstructionProgress(double, cv::Mat)), this, SLOT(reactToReconstructionProgressUpdate(double, cv::Mat)));
		QObject::connect(&_volume, SIGNAL(reconstructionFinished(cv::Mat, CtVolume::CompletionStatus)), this, SLOT(reactToReconstructionCompletion(cv::Mat, CtVolume::CompletionStatus)));
		QObject::connect(&_volume, SIGNAL(savingProgress(double)), this, SLOT(reactToSaveProgressUpdate(double)));
		QObject::connect(&_volume, SIGNAL(savingFinished(CtVolume::CompletionStatus)), this, SLOT(reactToSaveCompletion(CtVolume::CompletionStatus)));

		_inputFileEdit = new QLineEdit;
		_inputFileEdit->setPlaceholderText("Configuration File");
		QObject::connect(_inputFileEdit, SIGNAL(textChanged(QString)), this, SLOT(reactToTextChange(QString)));
		_browseButton = new QPushButton(tr("&Browse"));
		QObject::connect(_browseButton, SIGNAL(clicked()), this, SLOT(reactToBrowseButtonClick()));
		_completer = new QCompleter;
		QDirModel* model = new QDirModel(_completer);
		_completer->setModel(model);
		_inputFileEdit->setCompleter(_completer);

		_ramlakRadioButton = new QRadioButton(tr("R&am-Lak"));
		_ramlakRadioButton->setChecked(true);
		_shepploganRadioButton = new QRadioButton(tr("Sh&epp-Logan"));
		_hannRadioButton = new QRadioButton(tr("&Hann"));
		_filterLayout = new QVBoxLayout;
		_filterLayout->addWidget(_ramlakRadioButton);
		_filterLayout->addWidget(_shepploganRadioButton);
		_filterLayout->addWidget(_hannRadioButton);
		_filterGroupBox = new QGroupBox(tr("Filter Type"));
		_filterGroupBox->setLayout(_filterLayout);

		_xLabel = new QLabel("x:");
		_to1 = new QLabel("to");
		_xFrom = new QDoubleSpinBox;
		_xFrom->setRange(0, 1);
		_xFrom->setValue(0);
		_xFrom->setDecimals(3);
		_xFrom->setSingleStep(0.01);
		QObject::connect(_xFrom, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange(double)));
		_xTo = new QDoubleSpinBox;
		_xTo->setRange(0, 1);
		_xTo->setValue(1);
		_xTo->setDecimals(3);
		_xTo->setSingleStep(0.01);
		QObject::connect(_xTo, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange(double)));
		_xLayout = new QHBoxLayout;
		_xLayout->addWidget(_xLabel, 0);
		_xLayout->addWidget(_xFrom, 1);
		_xLayout->addWidget(_to1, 0);
		_xLayout->addWidget(_xTo, 1);
		_yLabel = new QLabel("y:");
		_to2 = new QLabel("to");
		_yFrom = new QDoubleSpinBox;
		_yFrom->setRange(0, 1);
		_yFrom->setValue(0);
		_yFrom->setDecimals(3);
		_yFrom->setSingleStep(0.01);
		QObject::connect(_yFrom, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange(double)));
		_yTo = new QDoubleSpinBox;
		_yTo->setRange(0, 1);
		_yTo->setValue(1);
		_yTo->setDecimals(3);
		_yTo->setSingleStep(0.01);
		QObject::connect(_yTo, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange(double)));
		_yLayout = new QHBoxLayout;
		_yLayout->addWidget(_yLabel, 0);
		_yLayout->addWidget(_yFrom, 1);
		_yLayout->addWidget(_to2, 0);
		_yLayout->addWidget(_yTo, 1);
		_zLabel = new QLabel("z:");
		_to3 = new QLabel("to");
		_zFrom = new QDoubleSpinBox;
		_zFrom->setRange(0, 1);
		_zFrom->setValue(0);
		_zFrom->setDecimals(3);
		_zFrom->setSingleStep(0.01);
		QObject::connect(_zFrom, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange(double)));
		_zTo = new QDoubleSpinBox;
		_zTo->setRange(0, 1);
		_zTo->setValue(1);
		_zTo->setDecimals(3);
		_zTo->setSingleStep(0.01);
		QObject::connect(_zTo, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange(double)));
		_zLayout = new QHBoxLayout;
		_zLayout->addWidget(_zLabel, 0);
		_zLayout->addWidget(_zFrom, 1);
		_zLayout->addWidget(_to3, 0);
		_zLayout->addWidget(_zTo, 1);
		_boundsLayout = new QVBoxLayout;
		_boundsLayout->addLayout(_xLayout);
		_boundsLayout->addLayout(_yLayout);
		_boundsLayout->addLayout(_zLayout);
		_boundsGroupBox = new QGroupBox(tr("Reconstruction bounds"));
		_boundsGroupBox->setLayout(_boundsLayout);

		_loadButton = new QPushButton(tr("&Load && Preprocess Images"));
		QObject::connect(_loadButton, SIGNAL(clicked()), this, SLOT(reactToLoadButtonClick()));
		_reconstructButton = new QPushButton(tr("&Reconstruct Volume"));
		QObject::connect(_reconstructButton, SIGNAL(clicked()), this, SLOT(reactToReconstructButtonClick()));
		_saveButton = new QPushButton(tr("&Save Volume"));
		QObject::connect(_saveButton, SIGNAL(clicked()), this, SLOT(reactToSaveButtonClick()));
		_runAllButton = new QPushButton(tr("R&un All and Save"));
		QObject::connect(_runAllButton, SIGNAL(clicked()), this, SLOT(reactToRunAllButtonClick()));
		_informationLabel = new QLabel;
		_statusLabel = new QLabel(tr("Load a configuration file"));

		_moreButton = new QPushButton(tr("&More..."));
		_moreMenu = new QMenu(_moreButton);
		_cmdAction = new QAction(tr("Save as &Batch File"), this);
		QObject::connect(_cmdAction, SIGNAL(triggered()), this, SLOT(reactToBatchFileAction()));
		_moreMenu->addAction(_cmdAction);
		_moreButton->setMenu(_moreMenu);
		QObject::connect(_moreMenu, SIGNAL(aboutToShow()), this, SLOT(adjustMenuWidth()));

		_leftLayout = new QVBoxLayout;
		_leftLayout->addStrut(250);
		_leftLayout->addWidget(_inputFileEdit);
		_leftLayout->addWidget(_browseButton, 0, Qt::AlignLeft);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_filterGroupBox);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_boundsGroupBox);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_loadButton);
		_leftLayout->addWidget(_reconstructButton);
		_leftLayout->addWidget(_saveButton);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_runAllButton);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_moreButton);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_informationLabel);
		_leftLayout->addSpacing(20);
		_leftLayout->addStretch(1);
		_leftLayout->addWidget(_statusLabel);

		_imageView = new hb::ImageView;

		_subLayout = new QHBoxLayout;
		_subLayout->addLayout(_leftLayout, 0);
		_subLayout->addWidget(_imageView, 1);

		_progressBar = new QProgressBar;
		_progressBar->setAlignment(Qt::AlignCenter);
		_stopButton = new QPushButton(tr("Stop"));
		QObject::connect(_stopButton, SIGNAL(clicked()), this, SLOT(reactToStopButtonClick()));

		_progressLayout = new QHBoxLayout;
		_progressLayout->addWidget(_progressBar, 1);
		_progressLayout->addWidget(_stopButton, 0);

		_mainLayout = new QVBoxLayout;
		_mainLayout->addLayout(_subLayout);
		_mainLayout->addLayout(_progressLayout);

		setLayout(_mainLayout);

		startupState();
	}

	MainInterface::~MainInterface() {
		delete _mainLayout;
		delete _subLayout;
		delete _leftLayout;
		delete _filterLayout;
		delete _boundsLayout;
		delete _progressLayout;
		delete _xLayout;
		delete _yLayout;
		delete _zLayout;
		delete _filterGroupBox;
		delete _ramlakRadioButton;
		delete _shepploganRadioButton;
		delete _hannRadioButton;
		delete _boundsGroupBox;
		delete _xFrom;
		delete _xTo;
		delete _yFrom;
		delete _yTo;
		delete _zFrom;
		delete _zTo;
		delete _xLabel;
		delete _yLabel;
		delete _zLabel;
		delete _to1;
		delete _to2;
		delete _to3;
		delete _inputFileEdit;
		delete _completer;
		delete _browseButton;
		delete _loadButton;
		delete _reconstructButton;
		delete _saveButton;
		delete _runAllButton;
		delete _moreButton;
		delete _stopButton;
		delete _moreMenu;
		delete _cmdAction;
		delete _progressBar;
		delete _imageView;
		delete _informationLabel;
		delete _statusLabel;
	}

	QSize MainInterface::sizeHint() const {
		return QSize(900, 600);
	}

	void MainInterface::dragEnterEvent(QDragEnterEvent* e) {
		if (e->mimeData()->hasUrls()) {
			if (!e->mimeData()->urls().isEmpty()) {
				e->acceptProposedAction();
			}
		}
	}

	void MainInterface::dropEvent(QDropEvent* e) {
		if (!e->mimeData()->urls().isEmpty()) {
			QString path = e->mimeData()->urls().first().toLocalFile();
			_inputFileEdit->setText(path);
			_inputFileEdit->setReadOnly(false);
			fileSelectedState();
		}
	}

	void MainInterface::keyPressEvent(QKeyEvent * e) {
		if (_sinogramDisplayActive) {
			if (e->key() == Qt::Key_Right) {
				setNextSinogramImage();
			} else if (e->key() == Qt::Key_Left) {
				setPreviousSinogramImage();
			} else {
				e->ignore();
				return;
			}
		} else if (_crossSectionDisplayActive || _reconstructionActive) {
			if (e->key() == Qt::Key_Up) {
				setNextSlice();
			} else if (e->key() == Qt::Key_Down) {
				setPreviousSlice();
			} else {
				e->ignore();
				return;
			}
		}
	}

	void MainInterface::disableAllControls() {
		_inputFileEdit->setEnabled(false);
		_browseButton->setEnabled(false);
		_loadButton->setEnabled(false);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(false);
		_cmdAction->setEnabled(false);
		_filterGroupBox->setEnabled(false);
		_boundsGroupBox->setEnabled(false);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(false);
		_stopButton->setEnabled(true);
	}

	void MainInterface::startupState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(false);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(false);
		_cmdAction->setEnabled(false);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(false);
		_imageView->resetImage();
		resetInfo();
		_stopButton->setEnabled(false);
	}

	void MainInterface::fileSelectedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(true);
		_cmdAction->setEnabled(true);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(false);
		_imageView->resetImage();
		_informationLabel->setText("<p>Estimated volume size: N/A</p><p>Volume dimensions: N/A</p><p>Sinogram size: N/A</p><p>Projections: N/A</p>");
		resetInfo();
		_stopButton->setEnabled(false);
	}

	void MainInterface::preprocessedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(true);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(true);
		_cmdAction->setEnabled(true);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = true;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(true);
		_stopButton->setEnabled(false);
	}

	void MainInterface::reconstructedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(true);
		_saveButton->setEnabled(true);
		_runAllButton->setEnabled(true);
		_cmdAction->setEnabled(true);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = true;
		_imageView->setRenderRectangle(false);
		_stopButton->setEnabled(false);
	}

	void MainInterface::setSinogramImage(size_t index) {
		if (index >= 0 && index < _volume.getSinogramSize()) {
			_currentIndex = index;
			_currentProjection = _volume.getProjectionAt(index);
			_currentProjection.image.convertTo(_currentProjection.image, CV_8U, 255);
			_imageView->setImage(_currentProjection.image);
			updateBoundsDisplay();
		}
	}

	void MainInterface::setNextSinogramImage() {
		size_t nextIndex = _currentIndex + 1;
		if (nextIndex >= _volume.getSinogramSize()) nextIndex = 0;
		setSinogramImage(nextIndex);
	}

	void MainInterface::setPreviousSinogramImage() {
		size_t previousIndex;
		if (_currentIndex == 0) {
			previousIndex = _volume.getSinogramSize() - 1;
		} else {
			previousIndex = _currentIndex - 1;
		}
		setSinogramImage(previousIndex);
	}

	void MainInterface::setSlice(size_t index) {
		if (index >= 0 && index < _volume.getZSize()) {
			_volume.setCrossSectionIndex(index);
			cv::Mat crossSection = _volume.getVolumeCrossSection(index);
			cv::normalize(crossSection, crossSection, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			_imageView->setImage(crossSection);
		}
	}

	void MainInterface::setNextSlice() {
		size_t nextSlice = _volume.getCrossSectionIndex() + 1;
		if (nextSlice >= _volume.getZSize()) nextSlice = _volume.getZSize() - 1;
		setSlice(nextSlice);
	}

	void MainInterface::setPreviousSlice() {
		size_t previousSlice;
		if (_volume.getCrossSectionIndex() != 0) {
			previousSlice = _volume.getCrossSectionIndex() - 1;
			setSlice(previousSlice);
		}
	}

	void MainInterface::updateBoundsDisplay() {
		double width = _volume.getImageWidth();
		double height = _volume.getImageHeight();
		double uOffset = _volume.getUOffset();
		double angleRad = (_currentProjection.angle / 180.0) * M_PI;
		double sine = sin(angleRad);
		double cosine = cos(angleRad);
		double xFrom = width*_xFrom->value() - width/2.0;
		double xTo = width*_xTo->value() - width / 2.0;
		double yFrom = width*_yFrom->value() - width / 2.0;
		double yTo = width*_yTo->value() - width / 2.0;
		double t1 = (-1)*xFrom*sine + yFrom*cosine + width / 2.0 + uOffset;
		double t2 = (-1)*xFrom*sine + yTo*cosine + width / 2.0 + uOffset;
		double t3 = (-1)*xTo*sine + yFrom*cosine + width / 2.0 + uOffset;
		double t4 = (-1)*xTo*sine + yTo*cosine + width / 2.0 + uOffset;
		double zFrom = height * _zFrom->value() + _currentProjection.heightOffset;
		double zTo = height * _zTo->value() + _currentProjection.heightOffset;
		double left = std::min({ t1, t2, t3, t4 });
		double right = std::max({ t1, t2, t3, t4 });
		_imageView->setRectangle(QRectF(left, height - zTo, right - left, zTo - zFrom));
	}

	void MainInterface::setStatus(QString text) {
		_statusLabel->setText(text);
	}

	void MainInterface::setInfo() {
		size_t xSize = _volume.getXSize();
		size_t ySize = _volume.getYSize();
		size_t zSize = _volume.getZSize();
		size_t width = _volume.getImageWidth();
		size_t height = _volume.getImageHeight();
		_informationLabel->setText("<p>" + tr("Estimated volume size: ") + QString::number(double(xSize*ySize*zSize) / 268435456.0, 'f', 2) + " Gb</p>"
								   "<p>" + tr("Volume dimensions: ") + QString::number(xSize) + "x" + QString::number(ySize) + "x" + QString::number(zSize) + "</p>"
								   "<p>" + tr("Sinogram size: ") + QString::number(double(width*height*_volume.getSinogramSize()) / 268435456.0, 'f', 2) + " Gb</p>"
								   "<p>" + tr("Projections: ") + QString::number(_volume.getSinogramSize()));
	}

	void MainInterface::resetInfo() {
		_informationLabel->setText("<p>Estimated volume size: N/A</p><p>Volume dimensions: N/A</p><p>Sinogram size: N/A</p><p>Projections: N/A</p>");
	}

	void MainInterface::reactToTextChange(QString text) {
		QFileInfo fileInfo(text);
		QMimeDatabase mime;
		if (text != "" && fileInfo.exists() && mime.mimeTypeForFile(fileInfo).inherits("text/plain")) {
			fileSelectedState();
			_inputFileEdit->setPalette(QPalette());
		} else {
			startupState();
			if (text != "") {
				QPalette palette;
				palette.setColor(QPalette::Text, Qt::red);
				_inputFileEdit->setPalette(palette);
			} else {
				_inputFileEdit->setPalette(QPalette());
			}
		}
	}

	void MainInterface::reactToBrowseButtonClick() {
		//QFileDialog dialog;
		//dialog.setNameFilter("Text Files (*.txt *.csv *.*);;");
		//dialog.setFileMode(QFileDialog::ExistingFile);
		//dialog.setWindowTitle("Open Config File");
		//dialog.setFilter(QDir::AllDirs);
		//dialog.exec();
		QString path = QFileDialog::getOpenFileName(this, tr("Open Config File"), QDir::rootPath(), "Text Files (*.txt *.csv *.*);;");

		if (!path.isEmpty()) {
			_inputFileEdit->setText(path);
		}
	}

	void MainInterface::reactToBoundsChange(double value) {
		if(_xFrom != QObject::sender()) _xFrom->setMaximum(_xTo->value());
		if (_xTo != QObject::sender()) _xTo->setMinimum(_xFrom->value());
		if (_yFrom != QObject::sender()) _yFrom->setMaximum(_yTo->value());
		if (_yTo != QObject::sender()) _yTo->setMinimum(_yFrom->value());
		if (_zFrom != QObject::sender()) _zFrom->setMaximum(_zTo->value());
		if (_zTo != QObject::sender()) _zTo->setMinimum(_zFrom->value());
		_volume.setVolumeBounds(_xFrom->value(), _xTo->value(), _yFrom->value(), _yTo->value(), _zFrom->value(), _zTo->value());
		if (_volume.getSinogramSize() > 0) {
			setInfo();
			updateBoundsDisplay();
		}
	}

	void MainInterface::reactToLoadButtonClick() {
		disableAllControls();
		setStatus(tr("Loding and preprocessing images..."));
		_timer.reset();
		FilterType type = FilterType::RAMLAK;
		if (_shepploganRadioButton->isChecked()) {
			type = FilterType::SHEPP_LOGAN;			
		} else if (_hannRadioButton->isChecked()) {
			type = FilterType::HANN;
		}
		std::thread(&CtVolume::sinogramFromImages, &_volume, _inputFileEdit->text().toStdString(), type).detach();	
	}

	void MainInterface::reactToReconstructButtonClick() {
		disableAllControls();
		_imageView->resetImage();
		setStatus(tr("Running backprojection..."));
		_timer.reset();
		_volume.setVolumeBounds(_xFrom->value(), _xTo->value(), _yFrom->value(), _yTo->value(), _zFrom->value(), _zTo->value());
		_reconstructionActive = true;
		std::thread(&CtVolume::reconstructVolume, &_volume).detach();
	}

	void MainInterface::reactToSaveButtonClick() {
		QString path;
		if (!_runAll) {
			path = QFileDialog::getSaveFileName(this, tr("Save Volume"), QDir::rootPath(), "Raw Files (*.raw);;");
			_savingPath = path;
		} else {
			path = _savingPath;
		}

		if (!path.isEmpty()) {
			disableAllControls();
			setStatus(tr("Writing volume to disk..."));
			_timer.reset();
			std::thread(&CtVolume::saveVolumeToBinaryFile, &_volume, path.toStdString()).detach();
		}
	}

	void MainInterface::reactToRunAllButtonClick() {
		_savingPath = QFileDialog::getSaveFileName(this, tr("Save Volume"), QDir::rootPath(), "Raw Files (*.raw);;");
		if (!_savingPath.isEmpty()) {
			_runAll = true;
			reactToLoadButtonClick();
		}
	}

	void MainInterface::reactToStopButtonClick() {
		_volume.stop();
		_stopButton->setEnabled(false);
		setStatus("Stopping...");
	}

	void MainInterface::adjustMenuWidth() {
		_moreMenu->setMinimumWidth(_moreButton->width());
	}

	void MainInterface::reactToBatchFileAction() {
#if defined Q_OS_WIN32
		//The strings for a windows system
		QString saveDialogCaption = tr("Create Batch File");
		QString saveDialogFiletype = "Command Line Scripts (*.cmd);;";
		QString promptWindowTitle = tr("Batch File Creation");
		QString status = tr("Batch file saved.");
		QString filepath = QFileDialog::getSaveFileName(this, saveDialogCaption, QDir::rootPath(), saveDialogFiletype);
#else
		//The strings for a linux system
		QString saveDialogCaption = tr("Create Shell Script");
		QString saveDialogFiletype = "Shell Scripts (*.sh);;";
		QString promptWindowTitle = tr("Shell Script Creation");
		QString status = tr("Shell script saved.");
		QString filepath = QFileDialog::getSaveFileName(this, saveDialogCaption, QDir::rootPath(), saveDialogFiletype);
#endif
		QDir cmdDir(QFileInfo(filepath).absoluteDir());
		if (!filepath.isEmpty()) {
			bool relativeExePath = false;
			bool relativeConfigPath = false;
			QMessageBox msgBox;
			msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
			msgBox.setWindowTitle(promptWindowTitle);
			msgBox.setText(tr("Shall the path to the <b>executable</b> be absolute or relative?"));
			msgBox.setButtonText(QMessageBox::Yes, tr("Use Relative Path"));
			msgBox.setButtonText(QMessageBox::No, tr("Use Absolute Path"));
			if (QMessageBox::Yes == msgBox.exec()) relativeExePath = true;
			msgBox.setText(tr("Shall the path to the <b>config file</b> be absolute or relative?"));
			if (QMessageBox::Yes == msgBox.exec()) relativeConfigPath = true;
			QString app = QCoreApplication::applicationFilePath();
			QString appPath;
			if (!relativeExePath) {
				appPath = app;
			} else {
				appPath = cmdDir.relativeFilePath(app);
			}
			QString configPath;
			if (!relativeConfigPath) {
				configPath = _inputFileEdit->text();
			} else {
				configPath = cmdDir.relativeFilePath(_inputFileEdit->text());
			}
			QFile file(filepath);
#if defined Q_OS_UNIX
			//On linux make it executable
			file.setPermissions(QFileDevice::ExeOther);
#endif
			if (file.open(QIODevice::ReadWrite | QIODevice::Truncate)) {
				QTextStream stream(&file);
#if defined Q_OS_UNIX
				stream << "#!/bin/sh" << ::endl;
#endif
				stream << "\"" << appPath << "\" -i \"" << configPath << "\" -o volume.raw";
				if (!_ramlakRadioButton->isChecked()) {
					if (_shepploganRadioButton->isChecked()) {
						stream << " -f shepplogan";
					} else {
						stream << " -f hann";
					}
				}
				if (_xFrom->value() != 0) stream << " --xmin " << _xFrom->value();
				if (_xTo->value() != 1) stream << " --xmax " << _xTo->value();
				if (_yFrom->value() != 0) stream << " --ymin " << _yFrom->value();
				if (_yTo->value() != 1) stream << " --ymax " << _yTo->value();
				if (_zFrom->value() != 0) stream << " --zmin " << _zFrom->value();
				if (_zTo->value() != 1) stream << " --zmax " << _zTo->value();
				file.close();
				setStatus(status);
			}
		}
	}

	void MainInterface::reactToLoadProgressUpdate(double percentage) {
		_progressBar->setValue(percentage);
	}

	void MainInterface::reactToLoadCompletion(CtVolume::CompletionStatus status) {
		_progressBar->reset();
		if (status.successful) {
			double time = _timer.getTime();
			setStatus(tr("Preprocessing finished (") + QString::number(time, 'f', 1) + "s).");
			setInfo();
			if (_runAll) {
				reactToReconstructButtonClick();
			} else {
				setSinogramImage(0);
				preprocessedState();
			}
		} else {
			if (!status.userInterrupted) {
				QMessageBox msgBox;
				msgBox.setText(status.errorMessage);
				msgBox.exec();
				setStatus(tr("Loading failed."));
			} else {
				setStatus(tr("Preprocessing stopped."));
			}
			if (_runAll) _runAll = false;
			fileSelectedState();
		}
	}

	void MainInterface::reactToReconstructionProgressUpdate(double percentage, cv::Mat crossSection) {
		_progressBar->setValue(percentage);
		if (percentage > 1.0) {
			double remaining = _timer.getTime() * ((100.0 - percentage) / percentage);
			int mins = std::floor(remaining / 60.0);
			int secs = std::floor(remaining - (mins * 60.0) + 0.5);
			setStatus(tr("Running backprojection... (app. %1:%2 min left)").arg(mins).arg(secs, 2, 10, QChar('0')));
		}
		cv::normalize(crossSection, crossSection, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		_imageView->setImage(crossSection);
	}

	void MainInterface::reactToReconstructionCompletion(cv::Mat crossSection, CtVolume::CompletionStatus status) {
		_reconstructionActive = false;
		_progressBar->reset();		
		if (status.successful) {
			cv::normalize(crossSection, crossSection, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			_imageView->setImage(crossSection);
			double time = _timer.getTime();
			setStatus(tr("Reconstruction finished (") + QString::number(time, 'f', 1) + "s).");
			if (_runAll) {
				reactToSaveButtonClick();
			} else {
				reconstructedState();
			}
		} else {
			if (!status.userInterrupted) {
				QMessageBox msgBox;
				msgBox.setText(status.errorMessage);
				msgBox.exec();
				setStatus(tr("Reconstruction failed."));
			} else {
				setStatus(tr("Reconstruction stopped."));
				setSinogramImage(0);
			}
			if (_runAll) _runAll = false;
			preprocessedState();
		}
	}

	void MainInterface::reactToSaveProgressUpdate(double percentage) {
		_progressBar->setValue(percentage);
	}

	void MainInterface::reactToSaveCompletion(CtVolume::CompletionStatus status) {
		_progressBar->reset();
		if (status.successful) {
			double time = _timer.getTime();
			setStatus(tr("Saving finished (") + QString::number(time, 'f', 1) + "s).");
		} else {
			if (!status.userInterrupted) {
				QMessageBox msgBox;
				msgBox.setText(status.errorMessage);
				msgBox.exec();
				setStatus(tr("Saving failed."));
			} else {
				setStatus(tr("Saving stopped."));
				QMessageBox msgBox;
				msgBox.setText(tr("The saving process was stopped. The file is probably unusable. Shall it be deleted?"));
				msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
				if (QMessageBox::Yes == msgBox.exec()) {
					QFile::remove(_savingPath);
				}
			}
		}
		_runAll = false;
		reconstructedState();
	}

}