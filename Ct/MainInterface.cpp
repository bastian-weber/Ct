#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent)
		: QWidget(parent),
		settings(new QSettings(QFileInfo(QCoreApplication::applicationFilePath()).absoluteDir().path() + "/ct.ini", QSettings::IniFormat)) {
		setAcceptDrops(true);

		this->volume.setEmitSignals(true);
		qRegisterMetaType<CompletionStatus>("CompletionStatus");
		QObject::connect(&this->volume, SIGNAL(loadingProgress(double)), this, SLOT(reactToLoadProgressUpdate(double)));
		QObject::connect(&this->volume, SIGNAL(loadingFinished(CompletionStatus)), this, SLOT(reactToLoadCompletion(CompletionStatus)));
		qRegisterMetaType<cv::Mat>("cv::Mat");
		QObject::connect(&this->volume, SIGNAL(reconstructionProgress(double, cv::Mat)), this, SLOT(reactToReconstructionProgressUpdate(double, cv::Mat)));
		QObject::connect(&this->volume, SIGNAL(reconstructionFinished(cv::Mat, CompletionStatus)), this, SLOT(reactToReconstructionCompletion(cv::Mat, CompletionStatus)));
		QObject::connect(&this->volume, SIGNAL(savingProgress(double)), this, SLOT(reactToSaveProgressUpdate(double)));
		QObject::connect(&this->volume, SIGNAL(savingFinished(CompletionStatus)), this, SLOT(reactToSaveCompletion(CompletionStatus)));

		this->inputFileEdit = new QLineEdit(this);
		this->inputFileEdit->setPlaceholderText("Configuration File");
		QObject::connect(this->inputFileEdit, SIGNAL(textChanged(QString)), this, SLOT(reactToTextChange(QString)));
		this->browseButton = new QPushButton(tr("&Browse"), this);
		QObject::connect(this->browseButton, SIGNAL(clicked()), this, SLOT(reactToBrowseButtonClick()));
		this->completer = new QCompleter(this);
		QDirModel* model = new QDirModel(this->completer);
		this->completer->setModel(model);
		this->inputFileEdit->setCompleter(this->completer);

		this->loadLayout = new QVBoxLayout;
		this->loadLayout->addWidget(this->inputFileEdit);
		this->loadLayout->addWidget(this->browseButton, 1, Qt::AlignLeft);
		this->loadGroupBox = new QGroupBox(tr("Configuration File"), this);
		this->loadGroupBox->setLayout(this->loadLayout);

		this->ramlakRadioButton = new QRadioButton(tr("R&am-Lak"), this);
		if (this->settings->value("filterType", "ramLak").toString() == "ramLak") this->ramlakRadioButton->setChecked(true);
		QObject::connect(this->ramlakRadioButton, SIGNAL(toggled(bool)), this, SLOT(saveFilterType()));
		this->shepploganRadioButton = new QRadioButton(tr("Sh&epp-Logan"), this);
		if (this->settings->value("filterType", "ramLak").toString() == "sheppLogan") this->shepploganRadioButton->setChecked(true);
		QObject::connect(this->shepploganRadioButton, SIGNAL(toggled(bool)), this, SLOT(saveFilterType()));
		this->hannRadioButton = new QRadioButton(tr("&Hann"), this);
		if (this->settings->value("filterType", "ramLak").toString() == "hann") this->hannRadioButton->setChecked(true);
		QObject::connect(this->hannRadioButton, SIGNAL(toggled(bool)), this, SLOT(saveFilterType()));
		this->filterLayout = new QVBoxLayout;
		this->filterLayout->addWidget(this->ramlakRadioButton);
		this->filterLayout->addWidget(this->shepploganRadioButton);
		this->filterLayout->addWidget(this->hannRadioButton);
		this->filterGroupBox = new QGroupBox(tr("Filter Type"), this);
		this->filterGroupBox->setLayout(this->filterLayout);

		this->xLabel = new QLabel("x:", this);
		this->xLabel->setStyleSheet("QLabel { color: red; }");
		this->to1 = new QLabel("to", this);
		this->xFrom = new QDoubleSpinBox(this);
		this->xFrom->setRange(0, 1);
		this->xFrom->setValue(this->settings->value("xFrom", 0).toDouble());
		this->xFrom->setDecimals(3);
		this->xFrom->setSingleStep(0.01);
		QObject::connect(this->xFrom, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange()));
		this->xTo = new QDoubleSpinBox(this);
		this->xTo->setRange(0, 1);
		this->xTo->setValue(this->settings->value("xTo", 1).toDouble());
		this->xTo->setDecimals(3);
		this->xTo->setSingleStep(0.01);
		QObject::connect(this->xTo, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange()));
		this->yLabel = new QLabel("y:", this);
		this->yLabel->setStyleSheet("QLabel { color: rgb(0, 160, 0); }");
		this->to2 = new QLabel("to", this);
		this->yFrom = new QDoubleSpinBox(this);
		this->yFrom->setRange(0, 1);
		this->yFrom->setValue(this->settings->value("yFrom", 0).toDouble());
		this->yFrom->setDecimals(3);
		this->yFrom->setSingleStep(0.01);
		QObject::connect(this->yFrom, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange()));
		this->yTo = new QDoubleSpinBox(this);
		this->yTo->setRange(0, 1);
		this->yTo->setValue(this->settings->value("yTo", 1).toDouble());
		this->yTo->setDecimals(3);
		this->yTo->setSingleStep(0.01);
		QObject::connect(this->yTo, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange()));
		this->zLabel = new QLabel("z:", this);
		this->zLabel->setStyleSheet("QLabel { color: blue; }");
		this->to3 = new QLabel("to", this);
		this->zFrom = new QDoubleSpinBox(this);
		this->zFrom->setRange(0, 1);
		this->zFrom->setValue(this->settings->value("zFrom", 0).toDouble());
		this->zFrom->setDecimals(3);
		this->zFrom->setSingleStep(0.01);
		QObject::connect(this->zFrom, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange()));
		this->zTo = new QDoubleSpinBox(this);
		this->zTo->setRange(0, 1);
		this->zTo->setValue(this->settings->value("zTo", 1).toDouble());
		this->zTo->setDecimals(3);
		this->zTo->setSingleStep(0.01);
		QObject::connect(this->zTo, SIGNAL(valueChanged(double)), this, SLOT(reactToBoundsChange()));
		this->resetButton = new QPushButton(tr("Reset All"), this);
		QObject::connect(this->resetButton, SIGNAL(clicked()), this, SLOT(resetBounds()));
		this->boundsLayout = new QGridLayout();
		this->boundsLayout->addWidget(this->xLabel, 0, 0);
		this->boundsLayout->addWidget(this->xFrom, 0, 1);
		this->boundsLayout->addWidget(this->to1, 0, 2);
		this->boundsLayout->addWidget(this->xTo, 0, 3);
		this->boundsLayout->addWidget(this->yLabel, 1, 0);
		this->boundsLayout->addWidget(this->yFrom, 1, 1);
		this->boundsLayout->addWidget(this->to2, 1, 2);
		this->boundsLayout->addWidget(this->yTo, 1, 3);
		this->boundsLayout->addWidget(this->zLabel, 2, 0);
		this->boundsLayout->addWidget(this->zFrom, 2, 1);
		this->boundsLayout->addWidget(this->to3, 2, 2);
		this->boundsLayout->addWidget(this->zTo, 2, 3);
		this->boundsLayout->addWidget(this->resetButton, 3, 1, 3, 1, Qt::AlignLeft);
		this->boundsLayout->setColumnStretch(0, 0);
		this->boundsLayout->setColumnStretch(1, 1);
		this->boundsLayout->setColumnStretch(2, 0);
		this->boundsLayout->setColumnStretch(3, 1);
		this->boundsGroupBox = new QGroupBox(tr("Reconstruction Bounds"), this);
		this->boundsGroupBox->setLayout(this->boundsLayout);

		this->cudaSettingsDialog = new CudaSettingsDialog(this->settings, volume.getCudaDeviceList(), this);
		QObject::connect(this->cudaSettingsDialog, SIGNAL(dialogConfirmed()), this, SLOT(updateInfo()));
		this->cudaGroupBox = new QGroupBox(tr("CUDA"), this);
		this->cudaCheckBox = new QCheckBox(tr("Use CUDA"), this->cudaGroupBox);
		QObject::connect(this->cudaCheckBox, SIGNAL(stateChanged(int)), this, SLOT(reactToCudaCheckboxChange()));
		this->cudaSettingsButton = new QPushButton(tr("CUDA Settings"), this->cudaGroupBox);
		QObject::connect(this->cudaSettingsButton, SIGNAL(clicked()), this->cudaSettingsDialog, SLOT(show()));
		this->cudaLayout = new QVBoxLayout();
		this->cudaLayout->addWidget(this->cudaCheckBox);
		this->cudaLayout->addWidget(this->cudaSettingsButton);
		this->cudaGroupBox->setLayout(this->cudaLayout);
		if (!volume.cudaAvailable()) {
			this->cudaCheckBox->setEnabled(false);
			this->cudaSettingsButton->setEnabled(false);
		}

		this->loadButton = new QPushButton(tr("&Load Configuration File"), this);
		QObject::connect(this->loadButton, SIGNAL(clicked()), this, SLOT(reactToLoadButtonClick()));
		this->reconstructButton = new QPushButton(tr("&Reconstruct Volume"), this);
		QObject::connect(this->reconstructButton, SIGNAL(clicked()), this, SLOT(reactToReconstructButtonClick()));
		this->saveButton = new QPushButton(tr("&Save Volume"), this);
		QObject::connect(this->saveButton, SIGNAL(clicked()), this, SLOT(reactToSaveButtonClick()));

		this->runAllButton = new QPushButton(tr("R&un All Steps and Save"), this);
		QObject::connect(this->runAllButton, SIGNAL(clicked()), this, SLOT(reactToRunAllButtonClick()));
		this->cmdButton = new QPushButton(tr("Save Current Settings as &Batch File"), this);
		QObject::connect(this->cmdButton, SIGNAL(clicked()), this, SLOT(reactToBatchFileAction()));
		this->advancedLayout = new QVBoxLayout;
		this->advancedLayout->addWidget(this->runAllButton);
		this->advancedLayout->addWidget(this->cmdButton);
		this->advancedGroupBox = new QGroupBox(tr("Advanced"), this);
		this->advancedGroupBox->setLayout(this->advancedLayout);

		this->informationLabel = new QLabel(this);
		this->infoLayout = new QVBoxLayout;
		this->infoLayout->addWidget(this->informationLabel);
		this->infoGroupBox = new QGroupBox(tr("Information"), this);
		this->infoGroupBox->setLayout(this->infoLayout);

		this->statusLabel = new QLabel(tr("Load a configuration file"), this);

		this->progressBar = new QProgressBar(this);
		this->progressBar->setAlignment(Qt::AlignCenter);
#ifdef Q_OS_WIN
		this->taskbarButton = new QWinTaskbarButton(this);
		this->taskbarProgress = this->taskbarButton->progress();
#endif
		this->stopButton = new QPushButton(tr("Stop"), this);
		QObject::connect(this->stopButton, SIGNAL(clicked()), this, SLOT(reactToStopButtonClick()));

		this->progressLayout = new QHBoxLayout();
		this->progressLayout->addWidget(this->progressBar, 1);
		this->progressLayout->addWidget(this->stopButton, 0);

		this->leftLayout = new QVBoxLayout;
		this->leftLayout->addStrut(250);
		this->leftLayout->addWidget(this->loadGroupBox);
		this->leftLayout->addSpacing(20);
		this->leftLayout->addWidget(this->filterGroupBox);
		this->leftLayout->addSpacing(20);
		this->leftLayout->addWidget(this->boundsGroupBox);
		this->leftLayout->addSpacing(20);
		this->leftLayout->addWidget(this->cudaGroupBox);
		this->leftLayout->addSpacing(20);
		this->leftLayout->addWidget(this->loadButton);
		this->leftLayout->addWidget(this->reconstructButton);
		this->leftLayout->addWidget(this->saveButton);
		this->leftLayout->addStretch(1);
		this->leftLayout->addLayout(this->progressLayout);
		this->leftLayout->addWidget(this->statusLabel);

		this->rightLayout = new QVBoxLayout;
		this->rightLayout->addStrut(250);
		this->rightLayout->addWidget(this->advancedGroupBox);
		this->rightLayout->addSpacing(20);
		this->rightLayout->addWidget(this->infoGroupBox);
		this->rightLayout->addStretch(1);

		this->imageView = new hb::ImageView(this);
		this->imageView->setExternalPostPaintFunction(this, &MainInterface::infoPaintFunction);

		this->subLayout = new QHBoxLayout;
		this->subLayout->addLayout(this->leftLayout, 0);
		this->subLayout->addWidget(this->imageView, 1);
		this->subLayout->addLayout(this->rightLayout, 0);

		setLayout(this->subLayout);

		this->startupState();
		this->inputFileEdit->setText(this->settings->value("last_path", "").toString());
		if(volume.cudaAvailable()) this->cudaCheckBox->setChecked(this->settings->value("useCuda", true).toBool());
		QSize lastSize = this->settings->value("size", QSize(-1, -1)).toSize();
		QPoint lastPos = this->settings->value("pos", QPoint(-1, -1)).toPoint();
		bool maximized = this->settings->value("maximized", false).toBool();

		//set volume bounds
		this->reactToBoundsChange();

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

	QSize MainInterface::sizeHint() const {
		return QSize(1053, 660);
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
		if (this->sinogramDisplayActive) {
			//draw projection number
			int digits = std::ceil(std::log10(this->volume.getSinogramSize()));
			canvas.drawText(QPoint(20, canvas.device()->height() - 15), QString("Projection %L1/%L2").arg(this->currentIndex, digits, 10, QChar('0')).arg(this->volume.getSinogramSize(), digits, 10, QChar('0')));
			//draw angle
			QString message = QString("%1 = %L2%3").arg(QChar(0x03B2)).arg(this->currentProjection.angle, 0, 'f', 2).arg(QChar(0x00B0));
			int textWidth = metrics.width(message);
			canvas.drawText(QPoint(canvas.device()->width() - 20 - textWidth, canvas.device()->height() - 15), message);
			//draw axes
			canvas.setBackgroundMode(Qt::TransparentMode);
			QPointF center(40, 40);
			double angleRad = this->currentProjection.angle*M_PI / 180.0;
			canvas.setPen(QPen(Qt::red, 2));
			QPointF xDelta(-20 * std::sin(angleRad), 10 * std::cos(angleRad));
			canvas.drawLine(center, center + xDelta);
			QString xString("x");
			QVector2D xVec(xDelta);
			xVec += xVec.normalized()*9;
			double xWidth = metrics.width(xString);
			canvas.drawText(center + xVec.toPointF() + QPointF(-xWidth / 2.0, 4), xString);
			canvas.setPen(QPen(QColor(0, 160, 0), 2));
			QPointF yDelta(20 * std::cos(angleRad), 10 * std::sin(angleRad));
			canvas.drawLine(center, center + yDelta);
			QString yString("y");
			QVector2D yVec(yDelta);
			yVec += yVec.normalized() * 9;
			double yWidth = metrics.width(yString);
			canvas.drawText(center + yVec.toPointF() + QPointF(-yWidth / 2.0, 4), yString);
			canvas.setPen(QPen(Qt::blue, 2));
			QPointF zDelta(0, -20);
			canvas.drawLine(center, center + zDelta);
			QString zString("z");
			QVector2D zVec(zDelta);
			zVec += zVec.normalized() * 9;
			double zWidth = metrics.width(zString);
			canvas.drawText(center + zVec.toPointF() + QPointF(-zWidth / 2.0, 4), zString);
			canvas.setPen(Qt::NoPen);
			canvas.setBrush(Qt::darkGray);
			canvas.drawEllipse(center, 3, 3);
		} else if (this->crossSectionDisplayActive || this->reconstructionActive || this->savingActive) {
			//draw slice number
			int digits = std::ceil(std::log10(this->volume.getCrossSectionSize()));
			canvas.drawText(QPoint(20, canvas.device()->height() - 15), QString("Slice %L1/%L2").arg(this->volume.getCrossSectionIndex(), digits, 10, QChar('0')).arg(this->volume.getCrossSectionSize(), digits, 10, QChar('0')));
			//draw axis name
			ct::Axis axis = this->volume.getCrossSectionAxis();
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
		if (e->mimeData()->hasUrls() && !this->controlsDisabled) {
			if (!e->mimeData()->urls().isEmpty()) {
				e->acceptProposedAction();
			}
		}
	}

	void MainInterface::dropEvent(QDropEvent* e) {
		if (!e->mimeData()->urls().isEmpty() && !this->controlsDisabled) {
			QString path = e->mimeData()->urls().first().toLocalFile();
			this->inputFileEdit->setText(path);
			this->inputFileEdit->setReadOnly(false);
			this->fileSelectedState();
		}
	}

	void MainInterface::keyPressEvent(QKeyEvent* e) {
		if (this->sinogramDisplayActive) {
			if (e->key() == Qt::Key_Right) {
				this->setNextSinogramImage();
			} else if (e->key() == Qt::Key_Left) {
				this->setPreviousSinogramImage();
			} else {
				e->ignore();
				return;
			}
		} else if (this->crossSectionDisplayActive || this->reconstructionActive || this->savingActive) {
			if (e->key() == Qt::Key_Up) {
				this->setPreviousSlice();
			} else if (e->key() == Qt::Key_Down) {
				this->setNextSlice();
			} else if (e->key() == Qt::Key_X) {
				this->volume.setCrossSectionAxis(Axis::X);
				this->setSlice(this->volume.getCrossSectionIndex());
			} else if (e->key() == Qt::Key_Y) {
				this->volume.setCrossSectionAxis(Axis::Y);
				this->setSlice(this->volume.getCrossSectionIndex());
			} else if (e->key() == Qt::Key_Z) {
				this->volume.setCrossSectionAxis(Axis::Z);
				this->setSlice(this->volume.getCrossSectionIndex());
			} else {
				e->ignore();
				return;
			}
		}
	}

	void MainInterface::wheelEvent(QWheelEvent* e) {
		if (this->crossSectionDisplayActive || this->reconstructionActive || this->savingActive) {
			if (e->modifiers() & Qt::AltModifier) {
				int signum = 1;
				if (e->delta() > 0) {
					signum = -1;
				}
				long nextSlice = this->volume.getCrossSectionIndex() + ((this->volume.getCrossSectionSize() / 10) * signum);
				if (nextSlice < 0) nextSlice = 0;
				if (nextSlice >= this->volume.getCrossSectionSize()) nextSlice = this->volume.getCrossSectionSize() - 1;
				this->setSlice(nextSlice);
				e->accept();
			} else {
				e->ignore();
			}
		} else if (this->sinogramDisplayActive) {
			if (e->modifiers() & Qt::AltModifier) {
				int signum = 1;
				if (e->delta() < 0) {
					signum = -1;
				}
				long nextProjection = this->currentIndex + ((this->volume.getSinogramSize() / 12) * signum);
				if (nextProjection < 0) nextProjection += this->volume.getSinogramSize();
				if (nextProjection >= this->volume.getSinogramSize()) nextProjection -= this->volume.getSinogramSize();
				this->setSinogramImage(nextProjection);
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
		this->taskbarButton->setWindow(this->windowHandle());
#endif
	}

	void MainInterface::closeEvent(QCloseEvent* e) {
		if (this->savingActive) {
			QMessageBox msgBox;
			msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
			msgBox.setWindowTitle(tr("Saving in progress"));
			msgBox.setText(tr("The application is still writing to the disk. Do you want to quit now or after saving finsihed?"));
			msgBox.setButtonText(QMessageBox::Yes, tr("Quit after saving"));
			msgBox.setButtonText(QMessageBox::No, tr("Quit now"));
			if (QMessageBox::Yes == msgBox.exec()) {
				//check if maybe now the saving is done
				if (!this->savingActive) {
					e->accept();
				} else {
					this->quitOnSaveCompletion = true;
					e->ignore();
				}
			} else {
				e->accept();
			}
			return;
		}
		this->settings->setValue("size", size());
		this->settings->setValue("pos", pos());
		this->settings->setValue("maximized", isMaximized());
		e->accept();
	}

	void MainInterface::disableAllControls() {
		this->inputFileEdit->setEnabled(false);
		this->browseButton->setEnabled(false);
		this->loadButton->setEnabled(false);
		this->reconstructButton->setEnabled(false);
		this->saveButton->setEnabled(false);
		this->runAllButton->setEnabled(false);
		this->cmdButton->setEnabled(false);
		this->filterGroupBox->setEnabled(false);
		this->boundsGroupBox->setEnabled(false);
		this->browseButton->setDefault(false);
		this->loadButton->setDefault(true);
		this->reconstructButton->setDefault(false);
		this->saveButton->setDefault(false);
		this->stopButton->setDefault(true);
		this->sinogramDisplayActive = false;
		this->crossSectionDisplayActive = false;
		this->controlsDisabled = true;
		this->imageView->setRenderRectangle(false);
		this->progressBar->setVisible(true);
		this->stopButton->setVisible(true);
		this->stopButton->setEnabled(true);
		this->cudaCheckBox->setEnabled(false);
		this->cudaSettingsButton->setEnabled(false);
		this->imageView->setFocus();
	}

	void MainInterface::startupState() {
		this->inputFileEdit->setEnabled(true);
		this->browseButton->setEnabled(true);
		this->loadButton->setEnabled(false);
		this->reconstructButton->setEnabled(false);
		this->saveButton->setEnabled(false);
		this->runAllButton->setEnabled(false);
		this->cmdButton->setEnabled(false);
		this->browseButton->setDefault(true);
		this->loadButton->setDefault(false);
		this->reconstructButton->setDefault(false);
		this->saveButton->setDefault(false);
		this->stopButton->setDefault(false);
		this->filterGroupBox->setEnabled(true);
		this->boundsGroupBox->setEnabled(true);
		this->sinogramDisplayActive = false;
		this->crossSectionDisplayActive = false;
		this->controlsDisabled = false;
		this->imageView->setRenderRectangle(false);
		this->imageView->resetImage();
		this->resetInfo();
		this->progressBar->setVisible(false);
		this->stopButton->setVisible(false);
		this->cudaCheckBox->setEnabled(this->volume.cudaAvailable());
		this->cudaSettingsButton->setEnabled(this->volume.cudaAvailable());
		this->browseButton->setFocus();
	}

	void MainInterface::fileSelectedState() {
		this->inputFileEdit->setEnabled(true);
		this->browseButton->setEnabled(true);
		this->loadButton->setEnabled(true);
		this->loadButton->setDefault(true);
		this->reconstructButton->setEnabled(false);
		this->saveButton->setEnabled(false);
		this->runAllButton->setEnabled(true);
		this->cmdButton->setEnabled(true);
		this->browseButton->setDefault(false);
		this->loadButton->setDefault(true);
		this->reconstructButton->setDefault(false);
		this->saveButton->setDefault(false);
		this->stopButton->setDefault(false);
		this->filterGroupBox->setEnabled(true);
		this->boundsGroupBox->setEnabled(true);
		this->sinogramDisplayActive = false;
		this->crossSectionDisplayActive = false;
		this->controlsDisabled = false;
		this->imageView->setRenderRectangle(false);
		this->imageView->resetImage();
		this->informationLabel->setText("<p>Memory required: N/A</p><p>Volume dimensions: N/A</p><p>Projections: N/A</p>");
		this->resetInfo();
		this->progressBar->setVisible(false);
		this->stopButton->setVisible(false);
		this->cudaCheckBox->setEnabled(this->volume.cudaAvailable());
		this->cudaSettingsButton->setEnabled(this->volume.cudaAvailable());
		this->loadButton->setFocus();
	}

	void MainInterface::preprocessedState() {
		this->inputFileEdit->setEnabled(true);
		this->browseButton->setEnabled(true);
		this->loadButton->setEnabled(true);
		this->reconstructButton->setEnabled(true);
		this->saveButton->setEnabled(false);
		this->runAllButton->setEnabled(true);
		this->cmdButton->setEnabled(true);
		this->browseButton->setDefault(false);
		this->loadButton->setDefault(false);
		this->reconstructButton->setDefault(true);
		this->saveButton->setDefault(false);
		this->stopButton->setDefault(false);
		this->filterGroupBox->setEnabled(true);
		this->boundsGroupBox->setEnabled(true);
		this->sinogramDisplayActive = true;
		this->crossSectionDisplayActive = false;
		this->controlsDisabled = false;
		this->imageView->setRenderRectangle(true);
		this->progressBar->setVisible(false);
		this->stopButton->setVisible(false);
		this->cudaCheckBox->setEnabled(this->volume.cudaAvailable());
		this->cudaSettingsButton->setEnabled(this->volume.cudaAvailable());
		this->imageView->setFocus();
	}

	void MainInterface::reconstructedState() {
		this->inputFileEdit->setEnabled(true);
		this->browseButton->setEnabled(true);
		this->loadButton->setEnabled(true);
		this->reconstructButton->setEnabled(true);
		this->saveButton->setEnabled(true);
		this->runAllButton->setEnabled(true);
		this->cmdButton->setEnabled(true);
		this->browseButton->setDefault(false);
		this->loadButton->setDefault(false);
		this->reconstructButton->setDefault(false);
		this->saveButton->setDefault(true);
		this->stopButton->setDefault(false);
		this->filterGroupBox->setEnabled(true);
		this->boundsGroupBox->setEnabled(true);
		this->sinogramDisplayActive = false;
		this->crossSectionDisplayActive = true;
		this->controlsDisabled = false;
		this->imageView->setRenderRectangle(false);
		this->progressBar->setVisible(false);
		this->stopButton->setVisible(false);
		this->cudaCheckBox->setEnabled(this->volume.cudaAvailable());
		this->cudaSettingsButton->setEnabled(this->volume.cudaAvailable());
		this->imageView->setFocus();
	}

	void MainInterface::setSinogramImage(size_t index) {
		if (index >= 0 && index < this->volume.getSinogramSize()) {
			this->currentIndex = index;
			this->currentProjection = this->volume.getProjectionAt(index);
			this->currentProjection.image.convertTo(this->currentProjection.image, CV_8U, 255);
			this->imageView->setImage(this->currentProjection.image);
			this->updateBoundsDisplay();
		}
	}

	void MainInterface::setNextSinogramImage() {
		size_t nextIndex = this->currentIndex + 1;
		if (nextIndex >= this->volume.getSinogramSize()) nextIndex = 0;
		this->setSinogramImage(nextIndex);
	}

	void MainInterface::setPreviousSinogramImage() {
		size_t previousIndex;
		if (this->currentIndex == 0) {
			previousIndex = this->volume.getSinogramSize() - 1;
		} else {
			previousIndex = this->currentIndex - 1;
		}
		this->setSinogramImage(previousIndex);
	}

	void MainInterface::setSlice(size_t index) {
		if (index >= 0 && index < this->volume.getCrossSectionSize()) {
			this->volume.setCrossSectionIndex(index);
			cv::Mat crossSection = this->volume.getVolumeCrossSection(this->volume.getCrossSectionAxis(), index);
			cv::Mat normalized;
			cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			this->imageView->setImage(normalized);
		}
	}

	void MainInterface::setNextSlice() {
		size_t nextSlice = this->volume.getCrossSectionIndex() + 1;
		if (nextSlice >= this->volume.getCrossSectionSize()) nextSlice = this->volume.getCrossSectionSize() - 1;
		this->setSlice(nextSlice);
	}

	void MainInterface::setPreviousSlice() {
		size_t previousSlice;
		if (this->volume.getCrossSectionIndex() != 0) {
			previousSlice = this->volume.getCrossSectionIndex() - 1;
			this->setSlice(previousSlice);
		}
	}

	void MainInterface::updateBoundsDisplay() {
		double width = this->volume.getImageWidth();
		double height = this->volume.getImageHeight();
		double uOffset = this->volume.getUOffset();
		double angleRad = (this->currentProjection.angle / 180.0) * M_PI;
		double sine = sin(angleRad);
		double cosine = cos(angleRad);
		double xFrom = width*this->xFrom->value() - width / 2.0;
		double xTo = width*this->xTo->value() - width / 2.0;
		double yFrom = width*this->yFrom->value() - width / 2.0;
		double yTo = width*this->yTo->value() - width / 2.0;
		double t1 = (-1)*xFrom*sine + yFrom*cosine + width / 2.0 + uOffset;
		double t2 = (-1)*xFrom*sine + yTo*cosine + width / 2.0 + uOffset;
		double t3 = (-1)*xTo*sine + yFrom*cosine + width / 2.0 + uOffset;
		double t4 = (-1)*xTo*sine + yTo*cosine + width / 2.0 + uOffset;
		double zFrom = height * this->zFrom->value() + this->currentProjection.heightOffset;
		double zTo = height * this->zTo->value() + this->currentProjection.heightOffset;
		double left = std::min({ t1, t2, t3, t4 });
		double right = std::max({ t1, t2, t3, t4 });
		this->imageView->setRectangle(QRectF(left, height - zTo, right - left, zTo - zFrom));
	}

	void MainInterface::setStatus(QString text) {
		this->statusLabel->setText(text);
	}

	void MainInterface::resetInfo() {
		this->informationLabel->setText("<p>Memory required: N/A</p><p>Volume dimensions: N/A</p><p>Projections: N/A</p>");
	}

	void MainInterface::setVolumeSettings() {
		FilterType type = FilterType::RAMLAK;
		if (this->shepploganRadioButton->isChecked()) {
			type = FilterType::SHEPP_LOGAN;
		} else if (this->hannRadioButton->isChecked()) {
			type = FilterType::HANN;
		}
		this->volume.setVolumeBounds(this->xFrom->value(), this->xTo->value(), this->yFrom->value(), this->yTo->value(), this->zFrom->value(), this->zTo->value());
		this->volume.setFrequencyFilterType(type);
		this->volume.setUseCuda(this->cudaCheckBox->isChecked());
		if(this->cudaCheckBox->isChecked()) this->volume.setActiveCudaDevices(this->cudaSettingsDialog->getActiveCudaDevices());
		this->volume.setGpuSpareMemory(this->settings->value("gpuSpareMemory", 200).toLongLong());
	}

	void MainInterface::reactToTextChange(QString text) {
		QFileInfo fileInfo(text);
		QMimeDatabase mime;
		if (text != "" && fileInfo.exists() && mime.mimeTypeForFile(fileInfo).inherits("text/plain")) {
			this->fileSelectedState();
			this->inputFileEdit->setPalette(QPalette());
			this->settings->setValue("last_path", text);
		} else {
			this->startupState();
			this->inputFileEdit->setFocus();
			if (text != "") {
				QPalette palette;
				palette.setColor(QPalette::Text, Qt::red);
				this->inputFileEdit->setPalette(palette);
			} else {
				this->inputFileEdit->setPalette(QPalette());
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
		QFileInfo dir(this->inputFileEdit->text());
		QString defaultPath;
		dir.exists() ? (dir.isDir() ? defaultPath = dir.filePath() : defaultPath = dir.path()) : QDir::rootPath();
		QString path = QFileDialog::getOpenFileName(this, tr("Open Config File"), defaultPath, "Text Files (*.txt *.csv *.*);;");

		if (!path.isEmpty()) {
			this->inputFileEdit->setText(path);
		}
	}

	void MainInterface::reactToBoundsChange() {
		if (this->xFrom != QObject::sender()) this->xFrom->setMaximum(this->xTo->value());
		if (this->xTo != QObject::sender()) this->xTo->setMinimum(this->xFrom->value());
		if (this->yFrom != QObject::sender()) this->yFrom->setMaximum(this->yTo->value());
		if (this->yTo != QObject::sender()) this->yTo->setMinimum(this->yFrom->value());
		if (this->zFrom != QObject::sender()) this->zFrom->setMaximum(this->zTo->value());
		if (this->zTo != QObject::sender()) this->zTo->setMinimum(this->zFrom->value());
		this->saveBounds();
		if (this->volume.getSinogramSize() > 0) {
			this->setVolumeSettings();
			this->updateInfo();
			this->updateBoundsDisplay();
		}
	}

	void MainInterface::reactToCudaCheckboxChange() {
		this->settings->setValue("useCuda", this->cudaCheckBox->isChecked());
		this->updateInfo();
	}

	void MainInterface::saveBounds() {
		this->settings->setValue("xFrom", this->xFrom->value());
		this->settings->setValue("xTo", this->xTo->value());
		this->settings->setValue("yFrom", this->yFrom->value());
		this->settings->setValue("yTo", this->yTo->value());
		this->settings->setValue("zFrom", this->zFrom->value());
		this->settings->setValue("zTo", this->zTo->value());
	}

	void MainInterface::resetBounds() {
		this->xFrom->setValue(0);
		this->xTo->setValue(1);
		this->yFrom->setValue(0);
		this->yTo->setValue(1);
		this->zFrom->setValue(0);
		this->zTo->setValue(1);
	}

	void MainInterface::saveFilterType() {
		if (this->ramlakRadioButton->isChecked()) {
			this->settings->setValue("filterType", "ramLak");
		} else if(this->shepploganRadioButton->isChecked()) {
			this->settings->setValue("filterType", "sheppLogan");
		} else {
			this->settings->setValue("filterType", "hann");
		}
	}

	void MainInterface::updateInfo() {
		if (this->volume.getSinogramSize() > 0) {
			this->setVolumeSettings();
			size_t xSize = this->volume.getXSize();
			size_t ySize = this->volume.getYSize();
			size_t zSize = this->volume.getZSize();
			double memory = double(this->volume.getRequiredMemoryUpperBound()) / 1024 / 1024 / 1024;
			QString infoText = tr("<p>Memory required: %L1Gb</p>"
								  "<p>Volume dimensions: %L2x%L3x%L4</p>"
								  "<p>Projections: %L5</p>");
			infoText = infoText.arg(memory, 0, 'f', 2).arg(xSize).arg(ySize).arg(zSize).arg(this->volume.getSinogramSize());
			this->informationLabel->setText(infoText);
		}
	}

	void MainInterface::reactToLoadButtonClick() {
		this->disableAllControls();
		this->setStatus(tr("Loading file and analysing images..."));
#ifdef Q_OS_WIN
		this->taskbarProgress->show();
#endif
		this->timer.reset();
		std::thread(&CtVolume::sinogramFromImages, &this->volume, this->inputFileEdit->text()).detach();
	}

	void MainInterface::reactToReconstructButtonClick() {
		this->disableAllControls();
		this->imageView->resetImage();
		this->setStatus(tr("Backprojecting..."));
#ifdef Q_OS_WIN
		this->taskbarProgress->show();
#endif
		this->timer.reset();
		this->predictionTimerSet = false;
		this->reconstructionActive = true;
		this->setVolumeSettings();
		std::thread(&CtVolume::reconstructVolume, &this->volume).detach();
	}

	void MainInterface::reactToSaveButtonClick() {
		QString path;
		if (!this->runAll) {
			path = QFileDialog::getSaveFileName(this, tr("Save Volume"), QDir::rootPath(), "Raw Files (*.raw);;");
			this->savingPath = path;
		} else {
			path = this->savingPath;
		}
		if (!path.isEmpty()) {
			this->disableAllControls();
			this->savingActive = true;
#ifdef Q_OS_WIN
			this->taskbarProgress->show();
#endif
			this->setStatus(tr("Writing volume to disk..."));
			this->timer.reset();
			std::thread(&CtVolume::saveVolumeToBinaryFile, &this->volume, path).detach();
		}
	}

	void MainInterface::reactToRunAllButtonClick() {
		this->savingPath = QFileDialog::getSaveFileName(this, tr("Save Volume"), QDir::rootPath(), "Raw Files (*.raw);;");
		if (!this->savingPath.isEmpty()) {
			this->runAll = true;
			this->reactToLoadButtonClick();
		}
	}

	void MainInterface::reactToStopButtonClick() {
		this->volume.stop();
		this->stopButton->setEnabled(false);
		this->setStatus("Stopping...");
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
				configPath = this->inputFileEdit->text();
			} else {
				configPath = cmdDir.relativeFilePath(this->inputFileEdit->text());
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
				if (!this->ramlakRadioButton->isChecked()) {
					if (this->shepploganRadioButton->isChecked()) {
						stream << " -f shepplogan";
					} else {
						stream << " -f hann";
					}
				}
				if (this->xFrom->value() != 0) stream << " --xmin " << this->xFrom->value();
				if (this->xTo->value() != 1) stream << " --xmax " << this->xTo->value();
				if (this->yFrom->value() != 0) stream << " --ymin " << this->yFrom->value();
				if (this->yTo->value() != 1) stream << " --ymax " << this->yTo->value();
				if (this->zFrom->value() != 0) stream << " --zmin " << this->zFrom->value();
				if (this->zTo->value() != 1) stream << " --zmax " << this->zTo->value();
				if (volume.cudaAvailable()) {
					if (!this->cudaCheckBox->isChecked()) stream << " -n";
					QStringList devices;
					std::vector<int> deviceIds = volume.getActiveCudaDevices();
					for (int& deviceId : deviceIds) {
						devices.append(QString::number(deviceId));
					}
					stream << " -d " << devices.join(',');
					stream << " -m " << this->settings->value("gpuSpareMemory", 200).toLongLong();
				}
				file.close();
				this->setStatus(status);
			}
		}
	}

	void MainInterface::reactToLoadProgressUpdate(double percentage) {
		this->progressBar->setValue(percentage);
#ifdef Q_OS_WIN
		this->taskbarProgress->setValue(percentage);
#endif
	}

	void MainInterface::reactToLoadCompletion(CompletionStatus status) {
		this->progressBar->reset();
#ifdef Q_OS_WIN
		this->taskbarProgress->hide();
		this->taskbarProgress->reset();
#endif
		if (status.successful) {
			double time = this->timer.getTime();
			this->setStatus(tr("Loading finished (") + QString::number(time, 'f', 1) + "s).");
			this->updateInfo();
			if (this->runAll) {
				this->reactToReconstructButtonClick();
			} else {
				this->setSinogramImage(0);
				this->preprocessedState();
			}
		} else {
			if (!status.userInterrupted) {
				QMessageBox msgBox;
				msgBox.setText(status.errorMessage);
				msgBox.exec();
				this->setStatus(tr("Loading failed."));
			} else {
				this->setStatus(tr("Loading stopped."));
			}
			if (this->runAll) this->runAll = false;
			this->fileSelectedState();
		}
	}

	void MainInterface::reactToReconstructionProgressUpdate(double percentage, cv::Mat crossSection) {
		this->progressBar->setValue(percentage);
#ifdef Q_OS_WIN
		this->taskbarProgress->setValue(percentage);
#endif
		if (percentage >= 1 && !this->predictionTimerSet) {
			this->predictionTimer.reset();
			this->predictionTimerSet = true;
		}
		if (percentage >= 3.0 && this->predictionTimerSet) {
			double remaining = double(this->predictionTimer.getTime()) * ((100.0 - percentage) / (percentage - 1.0));
			int mins = std::floor(remaining / 60.0);
			int secs = std::floor(remaining - (mins * 60.0) + 0.5);
			this->setStatus(tr("Backprojecting... (app. %1:%2 min left)").arg(mins).arg(secs, 2, 10, QChar('0')));
		}
		if (crossSection.data) {
			cv::Mat normalized;
			cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			this->imageView->setImage(normalized);
		}
	}

	void MainInterface::reactToReconstructionCompletion(cv::Mat crossSection, CompletionStatus status) {
		this->reconstructionActive = false;
		this->progressBar->reset();
#ifdef Q_OS_WIN
		this->taskbarProgress->hide();
		this->taskbarProgress->reset();
#endif
		if (status.successful) {
			cv::Mat normalized;
			cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			this->imageView->setImage(normalized);
			double time = this->timer.getTime();
			this->setStatus(tr("Reconstruction finished (") + QString::number(time, 'f', 1) + "s).");
			if (this->runAll) {
				this->reactToSaveButtonClick();
			} else {
				reconstructedState();
			}
		} else {
			if (!status.userInterrupted) {
				QMessageBox msgBox;
				msgBox.setText(status.errorMessage);
				msgBox.exec();
				this->setStatus(tr("Reconstruction failed."));
			} else {
				this->setStatus(tr("Reconstruction stopped."));
			}
			if (this->runAll) this->runAll = false;
			this->preprocessedState();
			this->setSinogramImage(0);
		}
	}

	void MainInterface::reactToSaveProgressUpdate(double percentage) {
		this->progressBar->setValue(percentage);
#ifdef Q_OS_WIN
		this->taskbarProgress->setValue(percentage);
#endif
	}

	void MainInterface::reactToSaveCompletion(CompletionStatus status) {
		this->savingActive = false;
		this->progressBar->reset();
#ifdef Q_OS_WIN
		this->taskbarProgress->hide();
		this->taskbarProgress->reset();
#endif
		if (status.successful) {
			double time = this->timer.getTime();
			this->setStatus(tr("Saving finished (") + QString::number(time, 'f', 1) + "s).");
		} else {
			if (!status.userInterrupted) {
				QMessageBox msgBox;
				msgBox.setText(status.errorMessage);
				msgBox.exec();
				this->setStatus(tr("Saving failed."));
			} else {
				this->setStatus(tr("Saving stopped."));
				this->askForDeletionOfIncompleteFile();
			}
		}
		this->runAll = false;
		reconstructedState();
		if (this->quitOnSaveCompletion) close();
	}

	void MainInterface::askForDeletionOfIncompleteFile() {
		QMessageBox msgBox;
		msgBox.setText(tr("The saving process was stopped. The file is probably unusable. Shall it be deleted?"));
		msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
		if (QMessageBox::Yes == msgBox.exec()) {
			QFileInfo fileInfo(this->savingPath);
			QString infoFileName = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".txt"));
			QFile::remove(this->savingPath);
			QFile::remove(infoFileName);
		}
	}

}