#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent) : QWidget(parent) {
		_mainLayout = new QHBoxLayout;

		_leftLayout = new QVBoxLayout;

		_openLayout = new QHBoxLayout;
		_inputFileEdit = new QLineEdit;
		_browseButton = new QPushButton(tr("&Browse"));
		_openLayout->addWidget(_inputFileEdit);
		_openLayout->addWidget(_browseButton);

		_loadButton = new QPushButton(tr("&Load && preprocess images"));

		_progressBar = new QProgressBar;
		_progressBar->setValue(0);
		_progressBar->setAlignment(Qt::AlignCenter);

		_leftLayout->addLayout(_openLayout);
		_leftLayout->addWidget(_loadButton);
		_leftLayout->addStretch(1);
		_leftLayout->addWidget(_progressBar);

		_mainLayout->addLayout(_leftLayout);

		setLayout(_mainLayout);
	}

	MainInterface::~MainInterface() {
		delete _mainLayout;
		delete _leftLayout;
		delete _openLayout;
		delete _inputFileEdit;
		delete _browseButton;
		delete _loadButton;
		delete _progressBar;
	}

	QSize MainInterface::sizeHint() const {
		return QSize(400, 700);
	}

}