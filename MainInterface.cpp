#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent) : QWidget(parent), _sinogramDisplayActive(false), _runAll(false) {
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
		_leftLayout->addWidget(_loadButton);
		_leftLayout->addWidget(_reconstructButton);
		_leftLayout->addWidget(_saveButton);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_runAllButton);
		_leftLayout->addSpacing(50);
		_leftLayout->addWidget(_informationLabel);
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
		delete _filterGroupBox;
		delete _ramlakRadioButton;
		delete _shepploganRadioButton;
		delete _hannRadioButton;
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
		_imageView->resetImage();
		_informationLabel->setText("");
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
		_imageView->resetImage();
		_informationLabel->setText("");
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
	}

	void MainInterface::setSinogramImage(size_t index) {
		if (index >= 0 && index < _volume.sinogramSize()) {
			_currentIndex = index;
			cv::Mat sinogramImage = _volume.sinogramImageAt(index);
			sinogramImage.convertTo(sinogramImage, CV_8U, 255);
			_imageView->setImage(sinogramImage);
		}
	}

	void MainInterface::setNextSinogramImage() {
		size_t nextIndex = _currentIndex + 1;
		if (nextIndex >= _volume.sinogramSize()) nextIndex = 0;
		setSinogramImage(nextIndex);
	}

	void MainInterface::setPreviousSinogramImage() {
		size_t previousIndex;
		if (_currentIndex == 0) {
			previousIndex = _volume.sinogramSize() - 1;
		} else {
			previousIndex = _currentIndex - 1;
		}
		setSinogramImage(previousIndex);
	}

	void MainInterface::setStatus(QString text) {
		_statusLabel->setText(text);
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
			_informationLabel->setText("<p>" + tr("Estimated volume size: ") + QString::number(double(_volume.getXSize()*_volume.getYSize()*_volume.getZSize()) / 268435456.0, 'f', 2) + " Gb</p>"
									   "<p>" + tr("Volume dimensions: ") + QString::number(_volume.getXSize()) + "x" + QString::number(_volume.getYSize()) + "x" + QString::number(_volume.getZSize()) + "</p>"
										"<p>" +tr("Sinogram size: ") + QString::number(double(_volume.getXSize()*_volume.getZSize()*_volume.sinogramSize()) / 268435456.0, 'f', 2) + " Gb</p>"
									   "<p>" + tr("Projections: ") + QString::number(_volume.sinogramSize()));
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