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
		_runAllButton = new QPushButton(tr("Run All and Save"));
		QObject::connect(_runAllButton, SIGNAL(clicked()), this, SLOT(reactToRunAllButtonClick()));
		_informationLabel = new QLabel;
		_statusLabel = new QLabel(tr("Load a configuration file"));

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

		_mainLayout = new QVBoxLayout;
		_mainLayout->addLayout(_subLayout);
		_mainLayout->addWidget(_progressBar);

		setLayout(_mainLayout);

		startupState();
	}

	MainInterface::~MainInterface() {
		delete _mainLayout;
		delete _subLayout;
		delete _leftLayout;
		delete _filterLayout;
		delete _boundsLayout;
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
		delete _browseButton;
		delete _loadButton;
		delete _reconstructButton;
		delete _saveButton;
		delete _runAllButton;
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
		_filterGroupBox->setEnabled(false);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(false);
	}

	void MainInterface::startupState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(false);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(false);
		_filterGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(false);
		_imageView->resetImage();
		resetInfo();
	}

	void MainInterface::fileSelectedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(true);
		_filterGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(false);
		_imageView->resetImage();
		_informationLabel->setText("<p>Estimated volume size: N/A</p><p>Volume dimensions: N/A</p><p>Sinogram size: N/A</p><p>Projections: N/A</p>");
		resetInfo();
	}

	void MainInterface::preprocessedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(true);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(true);
		_filterGroupBox->setEnabled(true);
		_sinogramDisplayActive = true;
		_crossSectionDisplayActive = false;
		_imageView->setRenderRectangle(true);
	}

	void MainInterface::reconstructedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(true);
		_saveButton->setEnabled(true);
		_runAllButton->setEnabled(true);
		_filterGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = true;
		_imageView->setRenderRectangle(false);
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
		if (_volume.getCrossSectionIndex() != 0) previousSlice = _volume.getCrossSectionIndex() - 1;
		setSlice(previousSlice);
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
		if (text != "") {
			fileSelectedState();
		} else {
			startupState();
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
			_inputFileEdit->setReadOnly(false);
			fileSelectedState();
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
		CtVolume::FilterType type = CtVolume::FilterType::RAMLAK;
		if (_shepploganRadioButton->isChecked()) {
			type = CtVolume::FilterType::SHEPP_LOGAN;			
		} else if (_hannRadioButton->isChecked()) {
			type = CtVolume::FilterType::HANN;
		}
		std::thread(&CtVolume::sinogramFromImages, &_volume, _inputFileEdit->text().toStdString(), type).detach();	
	}

	void MainInterface::reactToReconstructButtonClick() {
		disableAllControls();
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
			QMessageBox msgBox;
			msgBox.setText(status.errorMessage);
			msgBox.exec();
			setStatus(tr("Loading failed."));
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
			setStatus("Reconstruction finished (" + QString::number(time, 'f', 1) + "s).");
			if (_runAll) {
				reactToSaveButtonClick();
			} else {
				reconstructedState();
			}
		} else {
			QMessageBox msgBox;
			msgBox.setText(status.errorMessage);
			msgBox.exec();
			setStatus("Reconstruction failed.");
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
			setStatus("Saving finished (" + QString::number(time, 'f', 1) + "s).");
		} else {
			QMessageBox msgBox;
			msgBox.setText(status.errorMessage);
			msgBox.exec();
			setStatus("Saving failed.");
		}
		_runAll = false;
		reconstructedState();
	}

}