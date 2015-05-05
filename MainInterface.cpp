#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent) : QWidget(parent) {
		_openLabel = new QLabel(tr("Configuration file:"));
		_inputFileEdit = new QLineEdit;
		_inputFileEdit->setPlaceholderText("Configuration File");
		_browseButton = new QPushButton(tr("&Browse"));
		QObject::connect(_browseButton, SIGNAL(clicked()), this, SLOT(reactToBrowseButtonClick()));

		_openLayout = new QHBoxLayout;
		_openLayout->addWidget(_inputFileEdit);
		_openLayout->addWidget(_browseButton);

		_loadButton = new QPushButton(tr("&Load && Preprocess Images"));
		QObject::connect(_loadButton, SIGNAL(clicked()), this, SLOT(reactToLoadButtonClick()));
		_reconstructButton = new QPushButton(tr("&Reconstruct Volume"));
		QObject::connect(_reconstructButton, SIGNAL(clicked()), this, SLOT(reactToReconstructButtonClick()));
		_saveButton = new QPushButton(tr("&Save Volume"));

		_leftLayout = new QVBoxLayout;
		_leftLayout->addWidget(_openLabel);
		_leftLayout->addLayout(_openLayout);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_loadButton);
		_leftLayout->addWidget(_reconstructButton);
		_leftLayout->addWidget(_saveButton);
		_leftLayout->addStretch(1);

		_imageView = new hb::ImageView;

		_subLayout = new QHBoxLayout;
		_subLayout->addLayout(_leftLayout, 0);
		_subLayout->addWidget(_imageView, 1);

		_progressBar = new QProgressBar;
		_progressBar->setValue(0);
		_progressBar->setAlignment(Qt::AlignCenter);

		_mainLayout = new QVBoxLayout;
		_mainLayout->addLayout(_subLayout);
		_mainLayout->addWidget(_progressBar);

		setLayout(_mainLayout);
	}

	MainInterface::~MainInterface() {
		delete _mainLayout;
		delete _subLayout;
		delete _leftLayout;
		delete _openLayout;
		delete _openLabel;
		delete _inputFileEdit;
		delete _browseButton;
		delete _loadButton;
		delete _reconstructButton;
		delete _saveButton;
		delete _progressBar;
		delete _imageView;
	}

	QSize MainInterface::sizeHint() const {
		return QSize(700, 700);
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
			_inputFileEdit->insert(path);
			_inputFileEdit->setReadOnly(false);
		}
	}

	void MainInterface::reactToLoadButtonClick() {
		std::thread(&CtVolume::sinogramFromImages, &_volume, _inputFileEdit->text().toStdString(), CtVolume::RAMLAK).detach();	
	}

	void MainInterface::reactToReconstructButtonClick() {

	}

	void MainInterface::reactToLoadProgressUpdate() {

	}

	void MainInterface::reactToLoadCompletion() {

	}

	void MainInterface::reactToReconstructionProgressUpdate() {

	}

	void MainInterface::reactToReconstructionCompletion() {

	}

	void MainInterface::reactToSaveProgressUpdate() {

	}

	void MainInterface::reactToSaveCompletion() {

	}

}