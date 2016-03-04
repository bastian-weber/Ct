#ifndef CT_VIEWERINTERFACE
#define CT_VIEWERINTERFACE

#include <future>
#include <cmath>
#include <memory>

//OpenCV
#include <opencv2/highgui.hpp>

//Qt
#include <QtWidgets/QtWidgets>
#ifdef Q_OS_WIN
#include <QtWinExtras/QWinTaskbarProgress>
#include <QtWinExtras/QWinTaskbarButton>
#endif

#include "ImageView.h"
#include "Timer.h"
#include "Volume.h"

namespace ct {

	class ViewerInterface : public QWidget {
		Q_OBJECT
	public:
		ViewerInterface(QWidget *parent = 0);
		~ViewerInterface();
		QSize sizeHint() const;
		void infoPaintFunction(QPainter& canvas);
	protected:
		void dragEnterEvent(QDragEnterEvent* e);
		void dropEvent(QDropEvent* e);
		void keyPressEvent(QKeyEvent* e);
		void wheelEvent(QWheelEvent* e);
		void showEvent(QShowEvent* e);
		void closeEvent(QCloseEvent* e);
	private:
		void updateImage();
		void setNextSlice();
		void setPreviousSlice();
		size_t getCurrentSliceOfCurrentAxis() const;
		void setCurrentSliceOfCurrentAxis(size_t value);
		bool loadVolume(QString filename);

		Volume<float> volume;
		std::atomic<bool> volumeLoaded{ false };
		Axis currentAxis = Axis::Z;
		size_t currentSliceX = 0;
		size_t currentSliceY = 0;
		size_t currentSliceZ = 0;
		hb::Timer timer;
		std::shared_ptr<QSettings> settings;

		//interface widgets
		QHBoxLayout* mainLayout;
		hb::ImageView* imageView;
		//For the windows taskbar progress display
	#ifdef Q_OS_WIN
		QWinTaskbarButton* taskbarButton;
		QWinTaskbarProgress* taskbarProgress;
	#endif
	private slots:

	};

}

#endif