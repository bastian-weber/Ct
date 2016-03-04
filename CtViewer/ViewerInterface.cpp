#include "ViewerInterface.h"

namespace ct {

	ViewerInterface::ViewerInterface(QWidget *parent)
		: QWidget(parent),
		settings(new QSettings(QFileInfo(QCoreApplication::applicationFilePath()).absoluteDir().path() + "/ctviewer.ini", QSettings::IniFormat)) {
		setAcceptDrops(true);

		this->volume.setEmitSignals(true);
		qRegisterMetaType<CompletionStatus>("CompletionStatus");
		qRegisterMetaType<cv::Mat>("cv::Mat");

#ifdef Q_OS_WIN
		this->taskbarButton = new QWinTaskbarButton(this);
		this->taskbarProgress = this->taskbarButton->progress();
#endif
		this->setContentsMargins(0, 0, 0, 0);
		this->imageView = new hb::ImageView;
		this->imageView->setInterfaceBackgroundColor(Qt::black);
		this->imageView->setShowInterfaceOutline(false);
		this->imageView->setExternalPostPaintFunction(this, &ViewerInterface::infoPaintFunction);

		this->mainLayout = new QHBoxLayout();
		this->mainLayout->setContentsMargins(0, 0, 0, 0);
		this->mainLayout->addWidget(this->imageView);

		setLayout(this->mainLayout);

		QSize lastSize = this->settings->value("size", QSize(-1, -1)).toSize();
		QPoint lastPos = this->settings->value("pos", QPoint(-1, -1)).toPoint();
		bool maximized = this->settings->value("maximized", false).toBool();

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

	ViewerInterface::~ViewerInterface() {
		delete this->mainLayout;
		delete this->imageView;
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
		if (this->volumeLoaded) {
			if (e->key() == Qt::Key_Up) {
				this->setNextSlice();
			} else if (e->key() == Qt::Key_Down) {
				this->setPreviousSlice();
			} else if (e->key() == Qt::Key_X) {
				this->currentAxis = Axis::X;
				this->updateImage();
			} else if (e->key() == Qt::Key_Y) {
				this->currentAxis = Axis::Y;
				this->updateImage();
			} else if (e->key() == Qt::Key_Z) {
				this->currentAxis = Axis::Z;
				this->updateImage();
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
	}

	void ViewerInterface::closeEvent(QCloseEvent* e) {
		this->settings->setValue("size", size());
		this->settings->setValue("pos", pos());
		this->settings->setValue("maximized", isMaximized());
		e->accept();
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
		QFileInfo fileInfo(filename);
		QString infoFileName;
		if (fileInfo.suffix() == "txt") {
			infoFileName = filename;
			filename = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".raw"));
		} else {
			infoFileName = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".txt"));
		}
		if (!QFile::exists(filename)) {
			std::cout << "The volume file " << filename.toStdString() << " could not be found." << std::endl;
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
			//ask
		}
		if (!(xSize > 0 && ySize > 0 && zSize > 0)) {
			std::cout << "Volume could not be loaded because the volume dimensions are invalid." << std::endl;
			return false;
		}
		if (this->volume.loadFromBinaryFile<float>(filename.toStdString(), xSize, ySize, zSize)) {
			this->currentSliceX = this->volume.getSizeAlongDimension(Axis::X) / 2;
			this->currentSliceY = this->volume.getSizeAlongDimension(Axis::Y) / 2;
			this->currentSliceZ = this->volume.getSizeAlongDimension(Axis::Z) / 2;
			this->volumeLoaded = true;
			this->updateImage();
		} else {
			std::cout << "An error occured while trying to read the volume file." << std::endl;
			return false;
		}
		return true;
	}
}