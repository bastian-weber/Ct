#include "ViewerInterface.h"

namespace ct {

	ViewerInterface::ViewerInterface(QString const& openWithFilename, QWidget *parent)
		: QWidget(parent),
		settings(new QSettings(QFileInfo(QCoreApplication::applicationFilePath()).absoluteDir().path() + "/ctviewer.ini", QSettings::IniFormat)) {
		setAcceptDrops(true);

		//for "open with"
		this->openWithFilename = openWithFilename;
		QObject::connect(this, SIGNAL(windowLoaded()), this, SLOT(loadOpenWithFile()), Qt::ConnectionType(Qt::QueuedConnection | Qt::UniqueConnection));

		this->volume.setEmitSignals(true);
		qRegisterMetaType<CompletionStatus>("CompletionStatus");
		qRegisterMetaType<cv::Mat>("cv::Mat");
		QObject::connect(&this->volume, SIGNAL(loadingFinished(CompletionStatus)), this, SLOT(reactToLoadCompletion(CompletionStatus)));
		QObject::connect(&this->volume, SIGNAL(loadingProgress(double)), this, SLOT(reactToLoadProgressUpdate(double)));

		this->progressDialog = new QProgressDialog(tr("Reading volume from disk into RAM..."), tr("Stop"), 0, 100, this, Qt::CustomizeWindowHint | Qt::WindowTitleHint);
		QObject::connect(this->progressDialog, SIGNAL(canceled()), this, SLOT(stop()));
		this->progressDialog->setWindowModality(Qt::WindowModal);
		progressDialog->setMinimumWidth(350);
		progressDialog->setWindowTitle(tr("Loading..."));
		progressDialog->setMinimumDuration(500);
		progressDialog->reset();
#ifdef Q_OS_WIN
		this->taskbarButton = new QWinTaskbarButton(this);
		this->taskbarProgress = this->taskbarButton->progress();
#endif
		this->setContentsMargins(0, 0, 0, 0);

		this->imageView = new hb::ImageView;
		this->imageView->setInterfaceBackgroundColor(Qt::black);
		this->imageView->setShowInterfaceOutline(false);
		this->imageView->setExternalPostPaintFunction(this, &ViewerInterface::infoPaintFunction);
		this->imageView->setRightClickForHundredPercentView(false);

		this->mainLayout = new QHBoxLayout();
		this->mainLayout->setContentsMargins(0, 0, 0, 0);
		this->mainLayout->addWidget(this->imageView);

		setLayout(this->mainLayout);

		this->contextMenu = new QMenu(this);
		this->xAxisAction = new QAction(tr("X Axis"), this->contextMenu);
		this->xAxisAction->setCheckable(true);
		this->xAxisAction->setChecked(false);
		this->xAxisAction->setShortcut(Qt::Key_X);
		this->addAction(this->xAxisAction);
		QObject::connect(this->xAxisAction, SIGNAL(triggered()), this, SLOT(changeAxis()));
		this->yAxisAction = new QAction(tr("Y Axis"), this->contextMenu);
		this->yAxisAction->setCheckable(true);
		this->yAxisAction->setChecked(false);
		this->yAxisAction->setShortcut(Qt::Key_Y);
		this->addAction(this->yAxisAction);
		QObject::connect(this->yAxisAction, SIGNAL(triggered()), this, SLOT(changeAxis()));
		this->zAxisAction = new QAction(tr("Z Axis"), this->contextMenu);
		this->zAxisAction->setCheckable(true);
		this->zAxisAction->setChecked(true);
		this->zAxisAction->setShortcut(Qt::Key_Z);
		this->addAction(this->zAxisAction);
		QObject::connect(this->zAxisAction, SIGNAL(triggered()), this, SLOT(changeAxis()));
		this->axisActionGroup = new QActionGroup(this);
		this->axisActionGroup->addAction(xAxisAction);
		this->axisActionGroup->addAction(yAxisAction);
		this->axisActionGroup->addAction(zAxisAction);
		this->openDialogAction = new QAction(tr("Open Volume"), this->contextMenu);
		this->openDialogAction->setShortcut(Qt::CTRL + Qt::Key_O);
		this->addAction(this->openDialogAction);
		QObject::connect(this->openDialogAction, SIGNAL(triggered()), this, SLOT(openDialog()));
		this->saveImageAction = new QAction(tr("Save as Image"), this->contextMenu);
		this->saveImageAction->setShortcut(Qt::CTRL + Qt::Key_S);
		this->addAction(this->saveImageAction);
		QObject::connect(this->saveImageAction, SIGNAL(triggered()), this, SLOT(saveImageDialog()));

		this->contextMenu->addAction(this->openDialogAction);
		this->contextMenu->addSeparator();
		this->contextMenu->addAction(this->xAxisAction);
		this->contextMenu->addAction(this->yAxisAction);
		this->contextMenu->addAction(this->zAxisAction);
		this->contextMenu->addSeparator();
		this->contextMenu->addAction(this->saveImageAction);


		this->setContextMenuPolicy(Qt::CustomContextMenu);
		QObject::connect(this, SIGNAL(customContextMenuRequested(QPoint const&)), this, SLOT(showContextMenu(QPoint const&)));

		this->settingsDialog = new ImportSettingsDialog(this->settings, this);

		this->interfaceInitialState();

		QSize lastSize = this->settings->value("size", QSize(-1, -1)).toSize();
		QPoint lastPos = this->settings->value("pos", QPoint(-1, -1)).toPoint();
		bool maximized = this->settings->value("maximized", false).toBool();

		if (maximized) {
			setWindowState(Qt::WindowMaximized);
		} else {
			if (lastSize != QSize(-1, -1)) resize(lastSize);
			if (lastPos != QPoint(-1, -1)) move(lastPos);
		}
	}

	ViewerInterface::~ViewerInterface() {
		delete this->mainLayout;
		delete this->imageView;
		delete this->settingsDialog;
		delete this->contextMenu;
		delete this->axisActionGroup;
		delete this->xAxisAction;
		delete this->yAxisAction;
		delete this->zAxisAction;
		delete this->openDialogAction;
#ifdef Q_OS_WIN
		delete this->taskbarButton;
		delete this->taskbarProgress;
#endif
	}

	QSize ViewerInterface::sizeHint() const {
		return QSize(1053, 660);
	}

	void ViewerInterface::infoPaintFunction(QPainter& canvas) {
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
		if (this->volumeLoaded) {
			//draw slice number
			int digits = std::ceil(std::log10(this->volume.getSizeAlongDimension(this->currentAxis)));
			canvas.drawText(QPoint(20, canvas.device()->height() - 15), QString("Slice %L1/%L2").arg(this->getCurrentSliceOfCurrentAxis() + 1, digits, 10, QChar('0')).arg(this->volume.getSizeAlongDimension(this->currentAxis), digits, 10, QChar('0')));
			//draw axis name
			QString axisStr;
			switch (this->currentAxis) {
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

	void ViewerInterface::dragEnterEvent(QDragEnterEvent* e) {
		if (e->mimeData()->hasUrls()) {
			if (!e->mimeData()->urls().isEmpty()) {
				e->acceptProposedAction();
			}
		}
	}

	void ViewerInterface::dropEvent(QDropEvent* e) {
		if (!e->mimeData()->urls().isEmpty()) {
			QString path = e->mimeData()->urls().first().toLocalFile();
			this->loadVolume(path);
		}
	}

	void ViewerInterface::keyPressEvent(QKeyEvent* e) {
		if (e->key() == Qt::Key_Escape) {
			this->exitFullscreen();
			return;
		} else {
			if (this->volumeLoaded) {
				if (e->key() == Qt::Key_Up) {
					this->setNextSlice();
				} else if (e->key() == Qt::Key_Down) {
					this->setPreviousSlice();
				} else {
					e->ignore();
					return;
				}
			} else {
				e->ignore();
				return;
			}
		}

	}

	void ViewerInterface::wheelEvent(QWheelEvent* e) {
		if (this->volumeLoaded) {
			if (e->modifiers() & Qt::AltModifier) {
				int signum = 1;
				if (e->delta() < 0) {
					signum = -1;
				}
				long nextSlice = this->getCurrentSliceOfCurrentAxis() + ((this->volume.getSizeAlongDimension(this->currentAxis) / 10) * signum);
				if (nextSlice < 0) nextSlice = 0;
				if (nextSlice >= this->volume.getSizeAlongDimension(this->currentAxis)) nextSlice = this->volume.getSizeAlongDimension(this->currentAxis) - 1;
				this->setCurrentSliceOfCurrentAxis(nextSlice);
				this->updateImage();
				e->accept();
			} else {
				e->ignore();
			}
		} else {
			e->ignore();
		}
	}

	void ViewerInterface::showEvent(QShowEvent* e) {
#ifdef Q_OS_WIN
		this->taskbarButton->setWindow(this->windowHandle());
#endif
		emit(windowLoaded());
	}

	void ViewerInterface::closeEvent(QCloseEvent* e) {
		this->settings->setValue("size", size());
		this->settings->setValue("pos", pos());
		this->settings->setValue("maximized", isMaximized());
		e->accept();
	}

	void ViewerInterface::mouseDoubleClickEvent(QMouseEvent* e) {
		if (e->button() == Qt::LeftButton) {
			this->toggleFullscreen();
			e->accept();
		}
	}

	void ViewerInterface::changeEvent(QEvent* e) {
		if (e->type() == QEvent::WindowStateChange) {
			if (!this->isMinimized() && !this->isFullScreen()) {
				this->settings->setValue("maximized", this->isMaximized());
			} else if (this->isFullScreen()) {
				QWindowStateChangeEvent* windowStateChangeEvent = static_cast<QWindowStateChangeEvent*>(e);
				this->settings->setValue("maximized", bool(windowStateChangeEvent->oldState() & Qt::WindowMaximized));
			}
		}
	}

	void ViewerInterface::interfaceInitialState() {
		this->saveImageAction->setEnabled(false);
		this->xAxisAction->setEnabled(false);
		this->yAxisAction->setEnabled(false);
		this->zAxisAction->setEnabled(false);
	}

	void ViewerInterface::interfaceVolumeLoadedState() {
		this->saveImageAction->setEnabled(true);
		this->xAxisAction->setEnabled(true);
		this->yAxisAction->setEnabled(true);
		this->zAxisAction->setEnabled(true);
	}

	void ViewerInterface::updateImage() {
		if (this->getCurrentSliceOfCurrentAxis() < 0 || this->getCurrentSliceOfCurrentAxis() >= this->volume.getSizeAlongDimension(this->currentAxis)) {
			this->setCurrentSliceOfCurrentAxis(volume.getSizeAlongDimension(this->currentAxis) / 2);
		}
		cv::Mat crossSection = this->volume.getVolumeCrossSection(this->currentAxis, this->getCurrentSliceOfCurrentAxis());
		cv::Mat normalized;
		cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		this->imageView->setImage(normalized);
	}

	void ViewerInterface::setNextSlice() {
		size_t nextSlice = this->getCurrentSliceOfCurrentAxis() + 1;
		if (nextSlice >= this->volume.getSizeAlongDimension(this->currentAxis)) nextSlice = this->volume.getSizeAlongDimension(this->currentAxis) - 1;
		this->setCurrentSliceOfCurrentAxis(nextSlice);
		this->updateImage();
	}

	void ViewerInterface::setPreviousSlice() {
		size_t previousSlice;
		if (this->getCurrentSliceOfCurrentAxis() != 0) {
			previousSlice = this->getCurrentSliceOfCurrentAxis() - 1;
			this->setCurrentSliceOfCurrentAxis(previousSlice);
			this->updateImage();
		}
	}

	size_t ViewerInterface::getCurrentSliceOfCurrentAxis() const {
		switch (this->currentAxis) {
			case Axis::X:
				return this->currentSliceX;
			case Axis::Y:
				return this->currentSliceY;
			case Axis::Z:
				return this->currentSliceZ;
		}
	}

	void ViewerInterface::setCurrentSliceOfCurrentAxis(size_t value) {
		switch (this->currentAxis) {
			case Axis::X:
				this->currentSliceX = value;
				break;
			case Axis::Y:
				this->currentSliceY = value;
			case Axis::Z:
				this->currentSliceZ = value;
		}
	}
	bool ViewerInterface::loadVolume(QString filename) {
		if (this->loadingActive) return false;
		this->interfaceInitialState();
		this->loadingActive = true;
		this->reset();
		QFileInfo fileInfo(filename);
		QString infoFileName;
		if (fileInfo.suffix() == "txt") {
			infoFileName = filename;
			filename = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".raw"));
		} else {
			infoFileName = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".txt"));
		}
		if (!QFile::exists(filename)) {
			QMessageBox::critical(this, tr("Error"), QString("The volume file %1 could not be found.").arg(filename), QMessageBox::Close);
			this->loadingActive = false;
			return false;
		}
		bool informationFound = false;
		size_t xSize = 0;
		size_t ySize = 0;
		size_t zSize = 0;
		bool success;
		if (QFile::exists(infoFileName)) {
			informationFound = true;
			QFile file(infoFileName);
			if (file.open(QIODevice::ReadOnly)) {
				QTextStream in(&file);
				QString line;
				do {
					line = in.readLine();
					if (line.contains("X size:", Qt::CaseSensitive)) {
						QStringList parts = line.split('\t');
						for (QString& part : parts) {
							size_t parsed = part.toULongLong(&success);
							if (success) xSize = parsed;
						}
					} else if (line.contains("Y size:", Qt::CaseSensitive)) {
						QStringList parts = line.split('\t');
						for (QString& part : parts) {
							size_t parsed = part.toULongLong(&success);
							if (success) ySize = parsed;
						}
					} else if (line.contains("Z size:", Qt::CaseSensitive)) {
						QStringList parts = line.split('\t');
						for (QString& part : parts) {
							size_t parsed = part.toULongLong(&success);
							if (success) zSize = parsed;
						}
					}
				} while (!in.atEnd());
				if (!(xSize > 0 && ySize > 0 && zSize > 0)) informationFound = false;
			} else {
				informationFound = false;
			}
		}
		if (!QFile::exists(infoFileName) || !informationFound) {
			//ask for volume parameters
			QFileInfo volumeFileInfo(filename);
			if (this->settingsDialog->execForFilesize(volumeFileInfo.size()) != 0) {
				xSize = this->settingsDialog->getXSize();
				ySize = this->settingsDialog->getYSize();
				zSize = this->settingsDialog->getZSize();
			} else {
				this->loadingActive = false;
				return false;
			}
		}
		if (!(xSize > 0 && ySize > 0 && zSize > 0)) {
			QMessageBox::critical(this, tr("Error"), "Volume could not be loaded because the volume dimensions are invalid.", QMessageBox::Close);
			this->loadingActive = false;
			return false;
		}
		std::function<bool()> call = [=]() { 
			return this->volume.loadFromBinaryFile<float>(filename, xSize, ySize, zSize, QDataStream::SinglePrecision, QDataStream::LittleEndian); 
		};
		this->loadVolumeThread = std::async(std::launch::async, call);
		this->progressDialog->reset();
#ifdef Q_OS_WIN
		this->taskbarProgress->show();
#endif
		return true;
	}

	void ViewerInterface::reset() {
		this->volumeLoaded = false;
		this->imageView->resetImage();
		this->volume.clear();
	}

	void ViewerInterface::enterFullscreen() {
		this->showFullScreen();
	}

	void ViewerInterface::exitFullscreen() {
		if (this->settings->value("maximized", false).toBool()) {
			this->showMaximized();
		} else {
			this->showNormal();
		}
	}

	void ViewerInterface::toggleFullscreen() {
		if (this->isFullScreen()) {
			this->exitFullscreen();
		} else {
			this->enterFullscreen();
		}
	}

	void ViewerInterface::loadOpenWithFile() {
		if (!this->openWithFilename.isEmpty()) {
			this->loadVolume(openWithFilename);
			this->openWithFilename = QString();
		}
	}

	void ViewerInterface::reactToLoadProgressUpdate(double percentage) {
		this->progressDialog->setValue(percentage);
#ifdef Q_OS_WIN
		this->taskbarProgress->setValue(percentage);
#endif
	}

	void ViewerInterface::reactToLoadCompletion(CompletionStatus status) {
		loadVolumeThread.get();
		this->progressDialog->reset();
#ifdef Q_OS_WIN
		this->taskbarProgress->hide();
		this->taskbarProgress->reset();
#endif
		if (status.successful) {
			this->currentSliceX = this->volume.getSizeAlongDimension(Axis::X) / 2;
			this->currentSliceY = this->volume.getSizeAlongDimension(Axis::Y) / 2;
			this->currentSliceZ = this->volume.getSizeAlongDimension(Axis::Z) / 2;
			this->volumeLoaded = true;
			this->updateImage();
			this->interfaceVolumeLoadedState();
		} else {
			this->reset();
			if (!status.userInterrupted) {
				QMessageBox::critical(this, tr("Error"), status.errorMessage, QMessageBox::Close);
			}
		}
		this->loadingActive = false;
	}

	void ViewerInterface::stop() {
		this->volume.stop();
	}

	void ViewerInterface::showContextMenu(QPoint const& pos) {
		this->contextMenu->exec(mapToGlobal(pos));
	}

	void ViewerInterface::changeAxis() {
		if (this->axisActionGroup->checkedAction() == this->xAxisAction) {
			this->currentAxis = Axis::X;
		} else if (this->axisActionGroup->checkedAction() == this->yAxisAction) {
			this->currentAxis = Axis::Y;
		} else {
			this->currentAxis = Axis::Z;
		}
		this->updateImage();
	}

	void ViewerInterface::openDialog() {
		QString path = QFileDialog::getOpenFileName(this,
													tr("Open Volume or Volume Info File"),
													QDir::rootPath(),
													"Volume or Info Files (*.raw *.txt);;");

		if (!path.isEmpty()) {
			this->loadVolume(path);
		}
	}

	void ViewerInterface::saveImageDialog() {
		QString path = QFileDialog::getSaveFileName(this, tr("Save Image"), QDir::rootPath(), "Tif Files (*.tif);;");
		QMessageBox msgBox;
		msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
		msgBox.setWindowTitle(tr("Choose Colour Depth"));
		msgBox.setText(tr("Please choose the colour depth of the output image."));
		msgBox.setButtonText(QMessageBox::Yes, tr("16 bit"));
		msgBox.setButtonText(QMessageBox::No, tr("8 bit"));
		ImageBitDepth depth = ImageBitDepth::CHANNEL_16_BIT;
		if (QMessageBox::No == msgBox.exec()) {
			depth = ImageBitDepth::CHANNEL_8_BIT;
		}
		this->saveCurrentSliceAsImage(path, depth);
	}

	bool ViewerInterface::saveCurrentSliceAsImage(QString filename, ImageBitDepth bitDepth) {
		if (this->getCurrentSliceOfCurrentAxis() < 0 || this->getCurrentSliceOfCurrentAxis() >= this->volume.getSizeAlongDimension(this->currentAxis)) {
			return false;
		}
		cv::Mat crossSection = this->volume.getVolumeCrossSection(this->currentAxis, this->getCurrentSliceOfCurrentAxis());
		cv::Mat normalized;
		if (bitDepth == ImageBitDepth::CHANNEL_8_BIT) {
			cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		} else {
			cv::normalize(crossSection, normalized, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
		}
		try {
			std::vector<uchar> buffer;

			cv::imencode(".tif", normalized, buffer);
#ifdef Q_OS_WIN
			//wchar for utf-16
			std::ofstream file(filename.toStdWString(), std::iostream::binary);
#else
			//char for utf-8
			std::ifstream file(path.toStdString(), std::iostream::binary);
#endif
			if (!file.good()) {
				QMessageBox::critical(this, tr("Error"), tr("The image file could not be written. Maybe there is insufficient disk space or you are trying to overwrite a protected file."), QMessageBox::Close);
				return false;
			}
			char const* ptr = const_cast<char*>(reinterpret_cast<char*>(buffer.data()));
			file.write(ptr, buffer.size());
			file.close();
		} catch (...) {
			QMessageBox::critical(this, tr("Error"), tr("The image file could not be written. Maybe there is insufficient disk space or you are trying to overwrite a protected file."), QMessageBox::Close);
			return false;
		}
		return true;
	}

}