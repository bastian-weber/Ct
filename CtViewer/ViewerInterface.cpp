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

		this->volume.loadFromBinaryFile<float>("E:/Reconstructions/test.raw", 224, 224, 256);
		this->volumeLoaded = true;
		this->setCurrentSlice();
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
			canvas.drawText(QPoint(20, canvas.device()->height() - 15), QString("Slice %L1/%L2").arg(this->currentSlice, digits, 10, QChar('0')).arg(this->volume.getSizeAlongDimension(this->currentAxis), digits, 10, QChar('0')));
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
				this->setCurrentSlice();
			} else if (e->key() == Qt::Key_Y) {
				this->currentAxis = Axis::Y;
				this->setCurrentSlice();
			} else if (e->key() == Qt::Key_Z) {
				this->currentAxis = Axis::Z;
				this->setCurrentSlice();
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
				long nextSlice = this->currentSlice + ((this->volume.getSizeAlongDimension(this->currentAxis) / 10) * signum);
				if (nextSlice < 0) nextSlice = 0;
				if (nextSlice >= this->volume.getSizeAlongDimension(this->currentAxis)) nextSlice = this->volume.getSizeAlongDimension(this->currentAxis) - 1;
				this->currentSlice = nextSlice;
				this->setCurrentSlice();
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

	void ViewerInterface::setCurrentSlice() {
		if (this->currentSlice < 0 || this->currentSlice >= this->volume.getSizeAlongDimension(this->currentAxis)) {
			this->currentSlice = volume.getSizeAlongDimension(this->currentAxis) / 2;
		}
		cv::Mat crossSection = this->volume.getVolumeCrossSection(this->currentAxis, this->currentSlice);
		cv::Mat normalized;
		cv::normalize(crossSection, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		this->imageView->setImage(normalized);
	}

	void ViewerInterface::setNextSlice() {
		size_t nextSlice = this->currentSlice + 1;
		if (nextSlice >= this->volume.getSizeAlongDimension(this->currentAxis)) nextSlice = this->volume.getSizeAlongDimension(this->currentAxis) - 1;
		this->currentSlice = nextSlice;
		this->setCurrentSlice();
	}

	void ViewerInterface::setPreviousSlice() {
		size_t previousSlice;
		if (this->currentSlice != 0) {
			previousSlice = this->currentSlice - 1;
			this->currentSlice = previousSlice;
			this->setCurrentSlice();
		}
	}

}