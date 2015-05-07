#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent) : QWidget(parent), _sinogramDisplayActive(false) {
		setAcceptDrops(true);

		_volume.setEmitSignals(true);
		qRegisterMetaType<CtVolume::LoadStatus>("CtVolume::LoadStatus");
		QObject::connect(&_volume, SIGNAL(loadingProgress(double)), this, SLOT(reactToLoadProgressUpdate(double)));
		QObject::connect(&_volume, SIGNAL(loadingFinished(CtVolume::LoadStatus)), this, SLOT(reactToLoadCompletion(CtVolume::LoadStatus)));
		qRegisterMetaType<CtVolume::ReconstructStatus>("CtVolume::ReconstructStatus");
		qRegisterMetaType<cv::Mat>("cv::Mat");
		QObject::connect(&_volume, SIGNAL(reconstructionProgress(double, cv::Mat)), this, SLOT(reactToReconstructionProgressUpdate(double, cv::Mat)));
		QObject::connect(&_volume, SIGNAL(reconstructionFinished(CtVolume::ReconstructStatus, cv::Mat)), this, SLOT(reactToReconstructionCompletion(CtVolume::ReconstructStatus, cv::Mat)));
		qRegisterMetaType<CtVolume::SaveStatus>("CtVolume::SaveStatus");
		QObject::connect(&_volume, SIGNAL(savingProgress(double)), this, SLOT(reactToSaveProgressUpdate(double)));
		QObject::connect(&_volume, SIGNAL(savingFinished(CtVolume::SaveStatus)), this, SLOT(reactToSaveCompletion(CtVolume::SaveStatus)));

		_inputFileEdit = new QLineEdit;
		_inputFileEdit->setPlaceholderText("Configuration File");
		QObject::connect(_inputFileEdit, SIGNAL(textChanged(QString)), this, SLOT(reactToTextChange(QString)));
		_browseButton = new QPushButton(tr("&Browse"));
		QObject::connect(_browseButton, SIGNAL(clicked()), this, SLOT(reactToBrowseButtonClick()));

		_loadButton = new QPushButton(tr("&Load && Preprocess Images"));
		QObject::connect(_loadButton, SIGNAL(clicked()), this, SLOT(reactToLoadButtonClick()));
		_reconstructButton = new QPushButton(tr("&Reconstruct Volume"));
		QObject::connect(_reconstructButton, SIGNAL(clicked()), this, SLOT(reactToReconstructButtonClick()));
		_saveButton = new QPushButton(tr("&Save Volume"));
		QObject::connect(_saveButton, SIGNAL(clicked()), this, SLOT(reactToSaveButtonClick()));
		_informationLabel = new QLabel;
		_statusLabel = new QLabel;
		_statusLabel->setText("Load a configuration file");

		_leftLayout = new QVBoxLayout;
		_leftLayout->addStrut(250);
		_leftLayout->addWidget(_inputFileEdit);
		_leftLayout->addWidget(_browseButton, 0, Qt::AlignLeft);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_loadButton);
		_leftLayout->addWidget(_reconstructButton);
		_leftLayout->addWidget(_saveButton);
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
		delete _inputFileEdit;
		delete _browseButton;
		delete _loadButton;
		delete _reconstructButton;
		delete _saveButton;
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
		_sinogramDisplayActive = false;
	}

	void MainInterface::startupState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(false);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
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
		_sinogramDisplayActive = true;
	}

	void MainInterface::reconstructedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(true);
		_saveButton->setEnabled(true);
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
		setStatus("Loding and preprocessing images...");
		_timer.reset();
		std::thread(&CtVolume::sinogramFromImages, &_volume, _inputFileEdit->text().toStdString(), CtVolume::FilterType::RAMLAK).detach();	
	}

	void MainInterface::reactToReconstructButtonClick() {
		disableAllControls();
		setStatus("Running backprojection...");
		_timer.reset();
		std::thread(&CtVolume::reconstructVolume, &_volume).detach();
	}

	void MainInterface::reactToSaveButtonClick() {
		QString path = QFileDialog::getSaveFileName(this, tr("Save Volume"), QDir::rootPath(), "Raw Files (*.raw);;");

		if (!path.isEmpty()) {
			disableAllControls();
			setStatus("Writing volume to disk...");
			_timer.reset();
			std::thread(&CtVolume::saveVolumeToBinaryFile, &_volume, path.toStdString()).detach();
		}
	}

	void MainInterface::reactToLoadProgressUpdate(double percentage) {
		_progressBar->setValue(percentage);
	}

	void MainInterface::reactToLoadCompletion(CtVolume::LoadStatus status) {
		if (status == CtVolume::LoadStatus::SUCCESS) {
			_progressBar->reset();
			setSinogramImage(0);
			double time = _timer.getTime();
			setStatus("Preprocessing finished (" + QString::number(time, 'f', 1) + "s).");
			_informationLabel->setText("<p>Estimated volume size: " + QString::number(double(_volume.getXSize()*_volume.getYSize()*_volume.getZSize()) / 268435456.0, 'f', 2) + " Gb</p>"
									   "<p>Volume dimensions: " + QString::number(_volume.getXSize()) + "x" + QString::number(_volume.getYSize()) + "x" + QString::number(_volume.getZSize()) + "</p>"
									   "<p>Projections: " + QString::number(_volume.sinogramSize()));
			preprocessedState();
		}
	}

	void MainInterface::reactToReconstructionProgressUpdate(double percentage, cv::Mat crossSection) {
		_progressBar->setValue(percentage);
		cv::normalize(crossSection, crossSection, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		_imageView->setImage(crossSection);
	}

	void MainInterface::reactToReconstructionCompletion(CtVolume::ReconstructStatus status, cv::Mat crossSection) {
		if (status == CtVolume::ReconstructStatus::SUCCESS) {
			_progressBar->reset();
			cv::normalize(crossSection, crossSection, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			_imageView->setImage(crossSection);
			double time = _timer.getTime();
			setStatus("Reconstruction finished (" + QString::number(time, 'f', 1) + "s).");
			reconstructedState();
		}
	}

	void MainInterface::reactToSaveProgressUpdate(double percentage) {
		_progressBar->setValue(percentage);
	}

	void MainInterface::reactToSaveCompletion(CtVolume::SaveStatus status) {
		if (status == CtVolume::SaveStatus::SUCCESS) {
			_progressBar->reset();
			double time = _timer.getTime();
			setStatus("Saving finished (" + QString::number(time, 'f', 1) + "s).");
			reconstructedState();
		}
	}

}