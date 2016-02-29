#ifndef CT_MAININTERFACE
#define CT_MAININTERFACE

#include <future>
#include <cmath>
#include <memory>

//Qt
#include <QtWidgets/QtWidgets>
#ifdef Q_OS_WIN
#include <QtWinExtras/QWinTaskbarProgress>
#include <QtWinExtras/QWinTaskbarButton>
#endif

#include "ImageView.h"
#include "CudaSettingsDialog.h"
#include "CtVolume.h"
#include "Timer.h"

namespace ct {

	class MainInterface : public QWidget {
		Q_OBJECT
	public:
		MainInterface(QWidget *parent = 0);
		~MainInterface();
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
		void disableAllControls();
		void startupState();
		void fileSelectedState();
		void preprocessedState();
		void reconstructedState();
		void setSinogramImage(size_t index);
		void setNextSinogramImage();
		void setPreviousSinogramImage();
		void setSlice(size_t index);
		void setNextSlice();
		void setPreviousSlice();
		void updateBoundsDisplay();
		void setStatus(QString text);
		void resetInfo();
		void setVolumeSettings();

		CtVolume volume;
		std::atomic<bool> sinogramDisplayActive{ false };
		Projection currentProjection;
		std::atomic<bool> crossSectionDisplayActive{ false };
		std::atomic<bool> reconstructionActive{ false };
		std::atomic<bool> savingActive{ false };
		std::atomic<bool> quitOnSaveCompletion{ false };
		std::atomic<bool> controlsDisabled{ false };
		std::atomic<bool> runAll{ false };
		QString savingPath;
		size_t currentIndex;
		hb::Timer timer;
		hb::Timer predictionTimer;
		bool predictionTimerSet;
		std::shared_ptr<QSettings> settings;

		//interface widgets
		QHBoxLayout* subLayout;
		QVBoxLayout* leftLayout;
		QVBoxLayout* filterLayout;
		QGridLayout* boundsLayout;
		QVBoxLayout* cudaLayout;
		QHBoxLayout* progressLayout;
		QVBoxLayout* rightLayout;
		QVBoxLayout* loadLayout;
		QVBoxLayout* advancedLayout;
		QVBoxLayout* infoLayout;
		QGroupBox* loadGroupBox;
		QGroupBox* advancedGroupBox;
		QGroupBox* infoGroupBox;
		QGroupBox* filterGroupBox;
		QRadioButton* ramlakRadioButton;
		QRadioButton* shepploganRadioButton;
		QRadioButton* hannRadioButton;
		QGroupBox* boundsGroupBox;
		QDoubleSpinBox* xFrom;
		QDoubleSpinBox* xTo;
		QDoubleSpinBox* yFrom;
		QDoubleSpinBox* yTo;
		QDoubleSpinBox* zFrom;
		QDoubleSpinBox* zTo;
		QLabel* xLabel;
		QLabel* yLabel;
		QLabel* zLabel;
		QLabel* to1;
		QLabel* to2;
		QLabel* to3;
		QPushButton* resetButton;
		QGroupBox* cudaGroupBox;
		QCheckBox* cudaCheckBox;
		QPushButton* cudaSettingsButton;
		ct::CudaSettingsDialog* cudaSettingsDialog;
		QLineEdit* inputFileEdit;
		QCompleter* completer;
		QPushButton* browseButton;
		QPushButton* loadButton;
		QPushButton* reconstructButton;
		QPushButton* saveButton;
		QPushButton* runAllButton;
		QPushButton* cmdButton;
		QPushButton* stopButton;
		QProgressBar* progressBar;
		hb::ImageView* imageView;
		QLabel* informationLabel;
		QLabel* statusLabel;
		//For the windows taskbar progress display
	#ifdef Q_OS_WIN
		QWinTaskbarButton* taskbarButton;
		QWinTaskbarProgress* taskbarProgress;
	#endif
		private slots:
		void reactToTextChange(QString text);
		void reactToBrowseButtonClick();
		void reactToBoundsChange();
		void reactToCudaCheckboxChange();
		void saveBounds();
		void resetBounds();
		void saveFilterType();
		void updateInfo();
		void reactToLoadButtonClick();
		void reactToReconstructButtonClick();
		void reactToSaveButtonClick();
		void reactToRunAllButtonClick();
		void reactToStopButtonClick();
		void reactToBatchFileAction();
		void reactToLoadProgressUpdate(double percentage);
		void reactToLoadCompletion(CompletionStatus status);
		void reactToReconstructionProgressUpdate(double percentage, cv::Mat crossSection);
		void reactToReconstructionCompletion(cv::Mat crossSection, CompletionStatus status);
		void reactToSaveProgressUpdate(double percentage);
		void reactToSaveCompletion(CompletionStatus status);
		void askForDeletionOfIncompleteFile();
	};

}

#endif