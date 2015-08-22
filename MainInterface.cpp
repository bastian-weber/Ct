#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent)
		: QWidget(parent),
		_settings(QFileInfo(QCoreApplication::applicationFilePath()).absoluteDir().path() + "/ct.ini", QSettings::IniFormat) {
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

		_loadLayout = new QVBoxLayout;
		_loadLayout->addWidget(_inputFileEdit);
		_loadLayout->addWidget(_browseButton, 1, Qt::AlignLeft);
		_loadGroupBox = new QGroupBox(tr("Configuration File"));
		_loadGroupBox->setLayout(_loadLayout);

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
		_xLabel->setStyleSheet("QLabel { color: red; }");
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
		_yLabel->setStyleSheet("QLabel { color: rgb(0, 160, 0); }");
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
		_zLabel->setStyleSheet("QLabel { color: blue; }");
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
		_boundsGroupBox = new QGroupBox(tr("Reconstruction Bounds"));
		_boundsGroupBox->setLayout(_boundsLayout);

		_loadButton = new QPushButton(tr("&Load Config File"));
		QObject::connect(_loadButton, SIGNAL(clicked()), this, SLOT(reactToLoadButtonClick()));
		_reconstructButton = new QPushButton(tr("&Reconstruct Volume"));
		QObject::connect(_reconstructButton, SIGNAL(clicked()), this, SLOT(reactToReconstructButtonClick()));
		_saveButton = new QPushButton(tr("&Save Volume"));
		QObject::connect(_saveButton, SIGNAL(clicked()), this, SLOT(reactToSaveButtonClick()));

		_runAllButton = new QPushButton(tr("R&un All and Save"));
		QObject::connect(_runAllButton, SIGNAL(clicked()), this, SLOT(reactToRunAllButtonClick()));
		_cmdButton = new QPushButton(tr("Save as &Batch File"));
		QObject::connect(_cmdButton, SIGNAL(clicked()), this, SLOT(reactToBatchFileAction()));
		_advancedLayout = new QVBoxLayout;
		_advancedLayout->addWidget(_runAllButton);
		_advancedLayout->addWidget(_cmdButton);
		_advancedGroupBox = new QGroupBox(tr("Advanced"));
		_advancedGroupBox->setLayout(_advancedLayout);

		_informationLabel = new QLabel;
		_infoLayout = new QVBoxLayout;
		_infoLayout->addWidget(_informationLabel);
		_infoGroupBox = new QGroupBox(tr("Information"));
		_infoGroupBox->setLayout(_infoLayout);

		_statusLabel = new QLabel(tr("Load a configuration file"));

		_progressBar = new QProgressBar;
		_progressBar->setAlignment(Qt::AlignCenter);
#ifdef Q_OS_WIN
		_taskbarButton = new QWinTaskbarButton(this);
		_taskbarProgress = _taskbarButton->progress();
#endif
		_stopButton = new QPushButton(tr("Stop"));
		QObject::connect(_stopButton, SIGNAL(clicked()), this, SLOT(reactToStopButtonClick()));

		_progressLayout = new QHBoxLayout;
		_progressLayout->addWidget(_progressBar, 1);
		_progressLayout->addWidget(_stopButton, 0);

		_leftLayout = new QVBoxLayout;
		_leftLayout->addStrut(250);
		_leftLayout->addWidget(_loadGroupBox);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_filterGroupBox);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_boundsGroupBox);
		_leftLayout->addSpacing(20);
		_leftLayout->addWidget(_loadButton);
		_leftLayout->addWidget(_reconstructButton);
		_leftLayout->addWidget(_saveButton);
		_leftLayout->addStretch(1);
		_leftLayout->addLayout(_progressLayout);
		_leftLayout->addWidget(_statusLabel);

		_rightLayout = new QVBoxLayout;
		_rightLayout->addStrut(250);
		_rightLayout->addWidget(_advancedGroupBox);
		_rightLayout->addSpacing(20);
		_rightLayout->addWidget(_infoGroupBox);
		_rightLayout->addStretch(1);

		_imageView = new hb::ImageView;
		_imageView->setExternalPostPaintFunction(this, &MainInterface::infoPaintFunction);

		_subLayout = new QHBoxLayout;
		_subLayout->addLayout(_leftLayout, 0);
		_subLayout->addWidget(_imageView, 1);
		_subLayout->addLayout(_rightLayout, 0);

		setLayout(_subLayout);

		startupState();
		_inputFileEdit->setText(_settings.value("last_path", "").toString());
		QSize lastSize = _settings.value("size", QSize(-1, -1)).toSize();
		QPoint lastPos = _settings.value("pos", QPoint(-1, -1)).toPoint();
		bool maximized = _settings.value("maximized", false).toBool();

		//QPalette p(palette());
		//p.setColor(QPalette::Background, Qt::white);
		//setAutoFillBackground(true);
		//setPalette(p);
		if (maximized) {
			setWindowState(Qt::WindowMaximized);
		} else {
		if (lastSize != QSize(-1, -1)) resize(lastSize);
		if (lastPos != QPoint(-1, -1)) move(lastPos);
		}
	}

	MainInterface::~MainInterface() {
		delete _subLayout;
		delete _leftLayout;
		delete _filterLayout;
		delete _boundsLayout;
		delete _progressLayout;
		delete _xLayout;
		delete _yLayout;
		delete _zLayout;
		delete _rightLayout;
		delete _loadLayout;
		delete _advancedLayout;
		delete _infoLayout;
		delete _loadGroupBox;
		delete _advancedGroupBox;
		delete _infoGroupBox;
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
		delete _cmdButton;
		delete _stopButton;
		delete _progressBar;
		delete _imageView;
		delete _informationLabel;
		delete _statusLabel;
	#ifdef Q_OS_WIN
		delete _taskbarButton;
		delete _taskbarProgress;
	#endif
	}

	QSize MainInterface::sizeHint() const {
		return QSize(1053, 570);
	}

	void MainInterface::infoPaintFunction(QPainter& canvas) {
		canvas.setRenderHint(QPainter::Antialiasing, true);
		QPalette palette = qApp->palette();
		QPen textPen(palette.buttonText().color());
		canvas.setPen(textPen);
		canvas.setBrush(Qt::NoBrush);
		QFont font;
		font.setPointSize(10);
		canvas.setFont(font);
		QColor base = palette.base().color();
		base.setAlpha(200);
		canvas.setBackground(base);
		canvas.setBackgroundMode(Qt::OpaqueMode);
		QFontMetrics metrics(font);
		if (_sinogramDisplayActive) {
			//draw projection number
			int digits = std::ceil(std::log10(_volume.getSinogramSize()));
			canvas.drawText(QPoint(20, canvas.device()->height() - 15), QString("Projection %L1/%L2").arg(_currentIndex, digits, 10, QChar('0')).arg(_volume.getSinogramSize(), digits, 10, QChar('0')));
			//draw angle
			QString message = QString("%1 = %L2%3").arg(QChar(0x03B2)).arg(_currentProjection.angle, 0, 'f', 2).arg(QChar(0x00B0));
			int textWidth = metrics.width(message);
			canvas.drawText(QPoint(canvas.device()->width() - 20 - textWidth, canvas.device()->height() - 15), message);
			//draw axes
			canvas.setBackgroundMode(Qt::TransparentMode);
			QPointF center(30, 30);
			double angleRad = _currentProjection.angle*M_PI / 180.0;
			canvas.setPen(QPen(Qt::red, 2));
			QPointF xDelta(-20 * std::sin(angleRad), 10 * std::cos(angleRad));
			canvas.drawLine(center, center + xDelta);
			canvas.setPen(QPen(QColor(0, 160, 0), 2));
			QPointF yDelta(20 * std::cos(angleRad), 10 * std::sin(angleRad));
			canvas.drawLine(center, center + yDelta);
			canvas.setPen(QPen(Qt::blue, 2));
			QPointF zDelta(0, -20);
			canvas.drawLine(center, center + zDelta);
			canvas.setPen(Qt::NoPen);
			canvas.setBrush(Qt::darkGray);
			canvas.drawEllipse(center, 3, 3);
		} else if (_crossSectionDisplayActive || _reconstructionActive || _savingActive) {
			//draw slice number
			int digits = std::ceil(std::log10(_volume.getCrossSectionSize()));
			canvas.drawText(QPoint(20, canvas.device()->height() - 15), QString("Slice %L1/%L2").arg(_volume.getCrossSectionIndex(), digits, 10, QChar('0')).arg(_volume.getCrossSectionSize(), digits, 10, QChar('0')));
			//draw axis name
			ct::Axis axis = _volume.getCrossSectionAxis();
			QString axisStr;
			switch (axis) {
				case ct::Axis::X:
					axisStr = "X";
					break;
				case ct::Axis::Y:
					axisStr = "Y";
					break;
				case ct::Axis::Z:
					axisStr = "Z";
					break;
			}
			QString message = QString("%1-Axis").arg(axisStr);
			int textWidth = metrics.width(message);
			canvas.drawText(QPoint(canvas.device()->width() - 20 - textWidth, canvas.device()->height() - 15), message);
		}
	}

	void MainInterface::dragEnterEvent(QDragEnterEvent* e) {
		if (e->mimeData()->hasUrls() && !_controlsDisabled) {
			if (!e->mimeData()->urls().isEmpty()) {
				e->acceptProposedAction();
			}
		}
	}

	void MainInterface::dropEvent(QDropEvent* e) {
		if (!e->mimeData()->urls().isEmpty() && !_controlsDisabled) {
			QString path = e->mimeData()->urls().first().toLocalFile();
			_inputFileEdit->setText(path);
			_inputFileEdit->setReadOnly(false);
			fileSelectedState();
		}
	}

	void MainInterface::keyPressEvent(QKeyEvent* e) {
		if (_sinogramDisplayActive) {
			if (e->key() == Qt::Key_Right) {
				setNextSinogramImage();
			} else if (e->key() == Qt::Key_Left) {
				setPreviousSinogramImage();
			} else {
				e->ignore();
				return;
			}
		} else if (_crossSectionDisplayActive || _reconstructionActive || _savingActive) {
			if (e->key() == Qt::Key_Up) {
				setNextSlice();
			} else if (e->key() == Qt::Key_Down) {
				setPreviousSlice();
			} else if (e->key() == Qt::Key_X) {
				_volume.setCrossSectionAxis(Axis::X);
				setSlice(_volume.getCrossSectionIndex());
			} else if (e->key() == Qt::Key_Y) {
				_volume.setCrossSectionAxis(Axis::Y);
				setSlice(_volume.getCrossSectionIndex());
			} else if (e->key() == Qt::Key_Z) {
				_volume.setCrossSectionAxis(Axis::Z);
				setSlice(_volume.getCrossSectionIndex());
			} else {
				e->ignore();
				return;
			}
		}
	}

	void MainInterface::wheelEvent(QWheelEvent* e) {
		if (_crossSectionDisplayActive || _reconstructionActive || _savingActive) {
			if (e->modifiers() & Qt::ControlModifier) {
				int signum = 1;
				if (e->delta() < 0) {
					signum = -1;
				}
				long nextSlice = _volume.getCrossSectionIndex() + ((_volume.getCrossSectionSize() / 10) * signum);
				if (nextSlice < 0) nextSlice = 0;
				if (nextSlice >= _volume.getCrossSectionSize()) nextSlice = _volume.getCrossSectionSize() - 1;
				setSlice(nextSlice);
				e->accept();
			} else {
				e->ignore();
			}
		} else if (_sinogramDisplayActive) {
			if (e->modifiers() & Qt::ControlModifier) {
				int signum = 1;
				if (e->delta() < 0) {
					signum = -1;
				}
				long nextProjection = _currentIndex + ((_volume.getSinogramSize() / 12) * signum);
				if (nextProjection < 0) nextProjection += _volume.getSinogramSize();
				if (nextProjection >= _volume.getSinogramSize()) nextProjection -= _volume.getSinogramSize();
				setSinogramImage(nextProjection);
				e->accept();
			} else {
				e->ignore();
			}
		} else {
			e->ignore();
		}
	}

	void MainInterface::showEvent(QShowEvent* e) {
	#ifdef Q_OS_WIN
		_taskbarButton->setWindow(this->windowHandle());
	#endif
	}

	void MainInterface::closeEvent(QCloseEvent* e) {
		if (_savingActive) {
			QMessageBox msgBox;
			msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
			msgBox.setWindowTitle(tr("Saving in progress"));
			msgBox.setText(tr("The application is still writing to the disk. Do you want to quit now or after saving finsihed?"));
			msgBox.setButtonText(QMessageBox::Yes, tr("Quit after saving"));
			msgBox.setButtonText(QMessageBox::No, tr("Quit now"));
			if (QMessageBox::Yes == msgBox.exec()) {
				//check if maybe now the saving is done
				if (!_savingActive) {
					e->accept();
				} else {
					_quitOnSaveCompletion = true;
					e->ignore();
				}
			} else {
				e->accept();
			}
			return;
		}
		_settings.setValue("size", size());
		_settings.setValue("pos", pos());
		_settings.setValue("maximized", isMaximized());
		e->accept();
	}

	void MainInterface::disableAllControls() {
		_inputFileEdit->setEnabled(false);
		_browseButton->setEnabled(false);
		_loadButton->setEnabled(false);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(false);
		_cmdButton->setEnabled(false);
		_filterGroupBox->setEnabled(false);
		_boundsGroupBox->setEnabled(false);
		_browseButton->setDefault(false);
		_loadButton->setDefault(true);
		_reconstructButton->setDefault(false);
		_saveButton->setDefault(false);
		_stopButton->setDefault(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_controlsDisabled = true;
		_imageView->setRenderRectangle(false);
		_progressBar->setVisible(true);
		_stopButton->setVisible(true);
		_stopButton->setEnabled(true);
		_imageView->setFocus();
	}

	void MainInterface::startupState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(false);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(false);
		_cmdButton->setEnabled(false);
		_browseButton->setDefault(true);
		_loadButton->setDefault(false);
		_reconstructButton->setDefault(false);
		_saveButton->setDefault(false);
		_stopButton->setDefault(false);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_controlsDisabled = false;
		_imageView->setRenderRectangle(false);
		_imageView->resetImage();
		resetInfo();
		_progressBar->setVisible(false);
		_progressBar->setVisible(false);
		_browseButton->setFocus();
	}

	void MainInterface::fileSelectedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_loadButton->setDefault(true);
		_reconstructButton->setEnabled(false);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(true);
		_cmdButton->setEnabled(true);
		_browseButton->setDefault(false);
		_loadButton->setDefault(true);
		_reconstructButton->setDefault(false);
		_saveButton->setDefault(false);
		_stopButton->setDefault(false);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = false;
		_controlsDisabled = false;
		_imageView->setRenderRectangle(false);
		_imageView->resetImage();
		_informationLabel->setText("<p>Memory required: N/A</p><p>Volume dimensions: N/A</p><p>Projections: N/A</p>");
		resetInfo();
		_progressBar->setVisible(false);
		_stopButton->setVisible(false);
		_loadButton->setFocus();
	}

	void MainInterface::preprocessedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(true);
		_saveButton->setEnabled(false);
		_runAllButton->setEnabled(true);
		_cmdButton->setEnabled(true);
		_browseButton->setDefault(false);
		_loadButton->setDefault(false);
		_reconstructButton->setDefault(true);
		_saveButton->setDefault(false);
		_stopButton->setDefault(false);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = true;
		_crossSectionDisplayActive = false;
		_controlsDisabled = false;
		_imageView->setRenderRectangle(true);
		_progressBar->setVisible(false);
		_stopButton->setVisible(false);
		_imageView->setFocus();
	}

	void MainInterface::reconstructedState() {
		_inputFileEdit->setEnabled(true);
		_browseButton->setEnabled(true);
		_loadButton->setEnabled(true);
		_reconstructButton->setEnabled(true);
		_saveButton->setEnabled(true);
		_runAllButton->setEnabled(true);
		_cmdButton->setEnabled(true);
		_browseButton->setDefault(false);
		_loadButton->setDefault(false);
		_reconstructButton->setDefault(false);
		_saveButton->setDefault(true);
		_stopButton->setDefault(false);
		_filterGroupBox->setEnabled(true);
		_boundsGroupBox->setEnabled(true);
		_sinogramDisplayActive = false;
		_crossSectionDisplayActive = true;
		_controlsDisabled = false;
		_imageView->setRenderRectangle(false);
		_progressBar->setVisible(false);
		_stopButton->setVisible(false);
		_imageView->setFocus();
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
		if (index >= 0 && index < _volume.getCrossSectionSize()) {
			_volume.setCrossSectionIndex(index);
			cv::Mat crossSection = _volume.getVolumeCrossSection(index);
			cv::Mat normalized;
			cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			_imageView->setImage(normalized);
		}
	}

	void MainInterface::setNextSlice() {
		size_t nextSlice = _volume.getCrossSectionIndex() + 1;
		if (nextSlice >= _volume.getCrossSectionSize()) nextSlice = _volume.getCrossSectionSize() - 1;
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
		double xFrom = width*_xFrom->value() - width / 2.0;
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
		double memory = double(xSize*ySize*zSize + 2 * width*height) / 268435456.0;
		QString infoText = tr("<p>Memory required: %L1Gb</p>"
							  "<p>Volume dimensions: %L2x%L3x%L4</p>"
							  "<p>Projections: %L5</p>");
		infoText = infoText.arg(memory, 0, 'f', 2).arg(xSize).arg(ySize).arg(zSize).arg(_volume.getSinogramSize());
		_informationLabel->setText(infoText);
	}

	void MainInterface::resetInfo() {
		_informationLabel->setText("<p>Memory required: N/A</p><p>Volume dimensions: N/A</p><p>Projections: N/A</p>");
	}

	void MainInterface::reactToTextChange(QString text) {
		QFileInfo fileInfo(text);
		QMimeDatabase mime;
		if (text != "" && fileInfo.exists() && mime.mimeTypeForFile(fileInfo).inherits("text/plain")) {
			fileSelectedState();
			_inputFileEdit->setPalette(QPalette());
			_settings.setValue("last_path", text);
		} else {
			startupState();
			_inputFileEdit->setFocus();
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
		QFileInfo dir(_inputFileEdit->text());
		QString defaultPath;
		dir.exists() ? (dir.isDir() ? defaultPath = dir.filePath() : defaultPath = dir.path()) : QDir::rootPath();
		QString path = QFileDialog::getOpenFileName(this, tr("Open Config File"), defaultPath, "Text Files (*.txt *.csv *.*);;");

		if (!path.isEmpty()) {
			_inputFileEdit->setText(path);
		}
	}

	void MainInterface::reactToBoundsChange(double value) {
		if (_xFrom != QObject::sender()) _xFrom->setMaximum(_xTo->value());
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
		setStatus(tr("Loading file and analysing images..."));
	#ifdef Q_OS_WIN
		_taskbarProgress->show();
	#endif
		_timer.reset();
		std::thread(&CtVolume::sinogramFromImages, &_volume, _inputFileEdit->text().toStdString()).detach();
	}

	void MainInterface::reactToReconstructButtonClick() {
		disableAllControls();
		_imageView->resetImage();
		setStatus(tr("Backprojecting..."));
	#ifdef Q_OS_WIN
		_taskbarProgress->show();
	#endif
		_timer.reset();
		_volume.setVolumeBounds(_xFrom->value(), _xTo->value(), _yFrom->value(), _yTo->value(), _zFrom->value(), _zTo->value());
		FilterType type = FilterType::RAMLAK;
		if (_shepploganRadioButton->isChecked()) {
			type = FilterType::SHEPP_LOGAN;
		} else if (_hannRadioButton->isChecked()) {
			type = FilterType::HANN;
		}
		_reconstructionActive = true;
		std::thread(&CtVolume::reconstructVolume, &_volume, type).detach();
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
			_savingActive = true;
		#ifdef Q_OS_WIN
			_taskbarProgress->show();
		#endif
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
			if (!file.setPermissions(QFileDevice::ExeOther)) {
				std::cout << "Could not make shell script executable." << std::endl;
			}
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
	#ifdef Q_OS_WIN
		_taskbarProgress->setValue(percentage);
	#endif
	}

	void MainInterface::reactToLoadCompletion(CtVolume::CompletionStatus status) {
		_progressBar->reset();
	#ifdef Q_OS_WIN
		_taskbarProgress->hide();
		_taskbarProgress->reset();
	#endif
		if (status.successful) {
			double time = _timer.getTime();
			setStatus(tr("Loading finished (") + QString::number(time, 'f', 1) + "s).");
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
				setStatus(tr("Loading stopped."));
			}
			if (_runAll) _runAll = false;
			fileSelectedState();
		}
	}

	void MainInterface::reactToReconstructionProgressUpdate(double percentage, cv::Mat crossSection) {
		_progressBar->setValue(percentage);
	#ifdef Q_OS_WIN
		_taskbarProgress->setValue(percentage);
	#endif
		if (percentage > 1.0) {
			double remaining = _timer.getTime() * ((100.0 - percentage) / percentage);
			int mins = std::floor(remaining / 60.0);
			int secs = std::floor(remaining - (mins * 60.0) + 0.5);
			setStatus(tr("Backprojecting... (app. %1:%2 min left)").arg(mins).arg(secs, 2, 10, QChar('0')));
		}
		cv::Mat normalized;
		cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		_imageView->setImage(normalized);
	}

	void MainInterface::reactToReconstructionCompletion(cv::Mat crossSection, CtVolume::CompletionStatus status) {
		_reconstructionActive = false;
		_progressBar->reset();
	#ifdef Q_OS_WIN
		_taskbarProgress->hide();
		_taskbarProgress->reset();
	#endif
		if (status.successful) {
			cv::Mat normalized;
			cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			_imageView->setImage(normalized);
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
			}
			if (_runAll) _runAll = false;
			preprocessedState();
			setSinogramImage(0);
		}
	}

	void MainInterface::reactToSaveProgressUpdate(double percentage) {
		_progressBar->setValue(percentage);
	#ifdef Q_OS_WIN
		_taskbarProgress->setValue(percentage);
	#endif
	}

	void MainInterface::reactToSaveCompletion(CtVolume::CompletionStatus status) {
		_savingActive = false;
		_progressBar->reset();
	#ifdef Q_OS_WIN
		_taskbarProgress->hide();
		_taskbarProgress->reset();
	#endif
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
				askForDeletionOfIncompleteFile();
			}
		}
		_runAll = false;
		reconstructedState();
		if (_quitOnSaveCompletion) close();
	}

	void MainInterface::askForDeletionOfIncompleteFile() {
		QMessageBox msgBox;
		msgBox.setText(tr("The saving process was stopped. The file is probably unusable. Shall it be deleted?"));
		msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
		if (QMessageBox::Yes == msgBox.exec()) {
			QFile::remove(_savingPath);
		}
	}

}