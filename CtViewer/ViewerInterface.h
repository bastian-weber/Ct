#ifndef CT_VIEWERINTERFACE
#define CT_VIEWERINTERFACE

#include <future>
#include <cmath>
#include <memory>
#include <fstream>

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
#include "ImportSettingsDialog.h"

namespace ct {

	enum class ImageBitDepth {
		CHANNEL_8_BIT,
		CHANNEL_16_BIT
	};

	class ViewerInterface : public QWidget {
		Q_OBJECT
	public:
		ViewerInterface(QString const& openWithFilename = QString(), QWidget *parent = 0);
		QSize sizeHint() const;
		void infoPaintFunction(QPainter& canvas);
	protected:
		void dragEnterEvent(QDragEnterEvent* e);
		void dropEvent(QDropEvent* e);
		void keyPressEvent(QKeyEvent* e);
		void wheelEvent(QWheelEvent* e);
		void showEvent(QShowEvent* e);
		void closeEvent(QCloseEvent* e);
		void mouseDoubleClickEvent(QMouseEvent* e);
		void changeEvent(QEvent* e);
	private:
		void interfaceInitialState();
		void interfaceVolumeLoadedState();
		cv::Mat getNormalisedCrossSection() const;
		void updateImage();
		void setNextSlice();
		void setPreviousSlice();
		size_t getCurrentSliceOfCurrentAxis() const;
		void setCurrentSliceOfCurrentAxis(size_t value);
		bool loadVolume(QString filename);
		void reset();
		void enterFullscreen();
		void exitFullscreen();
		void toggleFullscreen();

		Volume<float> volume;
		float minValue = 1;
		float maxValue = 1;
		std::atomic<bool> volumeLoaded{ false };
		std::atomic<bool> loadingActive{ false };
		bool globalNormalisation = false;
		Axis currentAxis = Axis::Z;
		size_t currentSliceX = 0;
		size_t currentSliceY = 0;
		size_t currentSliceZ = 0;
		hb::Timer timer;
		std::shared_ptr<QSettings> settings;
		std::future<bool> loadVolumeThread;
		QString openWithFilename;

		//interface widgets
		QHBoxLayout* mainLayout;
		hb::ImageView* imageView;
		QProgressDialog* progressDialog;
		ImportSettingsDialog* settingsDialog;
		QMenu* contextMenu;
		QActionGroup* axisActionGroup;
		QActionGroup* normActionGroup;
		QAction* xAxisAction;
		QAction* yAxisAction;
		QAction* zAxisAction;
		QAction* openDialogAction;
		QAction* saveImageAction;
		QAction* localNormAction;
		QAction* globalNormAction;
		//For the windows taskbar progress display
#ifdef Q_OS_WIN
		QWinTaskbarButton* taskbarButton;
		QWinTaskbarProgress* taskbarProgress;
#endif
	private slots:
		void loadOpenWithFile();
		void reactToLoadProgressUpdate(double percentage);
		void reactToLoadCompletion(CompletionStatus status);
		void stop();
		void showContextMenu(QPoint const& pos);
		void changeAxis();
		void changeNormalisation();
		void openDialog();
		void saveImageDialog();
		bool saveCurrentSliceAsImage(QString filename, ImageBitDepth bitDepth);
	signals:
		void windowLoaded();
		void progressUpdate(int progress) const;
	};

}

#endif