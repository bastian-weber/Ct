#ifndef CT_MAININTERFACE
#define CT_MAININTERFACE

#include <future>

//Qt
#include <QtWidgets/QtWidgets>
#include <QtWinExtras/QWinTaskbarProgress>
#include <QtWinExtras/QWinTaskbarButton>

#include "ImageView.h"
#include "CtVolume.h"
#include "Timer.h"

namespace ct {

	class MainInterface : public QWidget{
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
		void setInfo();
		void resetInfo();

		CtVolume _volume;
		std::atomic<bool> _sinogramDisplayActive;
		Projection _currentProjection;
		std::atomic<bool> _crossSectionDisplayActive;
		std::atomic<bool> _reconstructionActive;
		std::atomic<bool> _savingActive;
		std::atomic<bool> _quitOnSaveCompletion;
		std::atomic<bool> _controlsDisabled;
		std::atomic<bool> _runAll;
		QString _savingPath;
		size_t _currentIndex;
		hb::Timer _timer;

		//interface widgets
		QVBoxLayout* _mainLayout;
		QHBoxLayout* _subLayout;
		QVBoxLayout* _leftLayout;
		QHBoxLayout* _loadButtonLayout;
		QVBoxLayout* _filterLayout;
		QVBoxLayout* _boundsLayout;
		QHBoxLayout* _progressLayout;
		QHBoxLayout* _xLayout;
		QHBoxLayout* _yLayout;
		QHBoxLayout* _zLayout;
		QVBoxLayout* _rightLayout;
		QVBoxLayout* _loadLayout;
		QVBoxLayout* _reconstructLayout;
		QVBoxLayout* _saveLayout;
		QVBoxLayout* _advancedLayout;
		QVBoxLayout* _infoLayout;
		QGroupBox* _loadGroupBox;
		QGroupBox* _reconstructGroupBox;
		QGroupBox* _saveGroupBox;
		QGroupBox* _advancedGroupBox;
		QGroupBox* _infoGroupBox;
		QGroupBox* _filterGroupBox;
		QRadioButton* _ramlakRadioButton;
		QRadioButton* _shepploganRadioButton;
		QRadioButton* _hannRadioButton;
		QGroupBox* _boundsGroupBox;
		QDoubleSpinBox* _xFrom;
		QDoubleSpinBox* _xTo;
		QDoubleSpinBox* _yFrom;
		QDoubleSpinBox* _yTo;
		QDoubleSpinBox* _zFrom;
		QDoubleSpinBox* _zTo;
		QLabel* _xLabel;
		QLabel* _yLabel;
		QLabel* _zLabel;
		QLabel* _to1;
		QLabel* _to2;
		QLabel* _to3;
		QLineEdit* _inputFileEdit;
		QCompleter* _completer;
		QPushButton* _browseButton;
		QPushButton* _loadButton;
		QPushButton* _reconstructButton;
		QPushButton* _saveButton;
		QPushButton* _runAllButton;
		QPushButton* _cmdButton;
		QPushButton* _stopButton;
		QProgressBar* _progressBar;
		QWinTaskbarButton* _taskbarButton;
		QWinTaskbarProgress* _taskbarProgress;
		hb::ImageView* _imageView;
		QLabel* _informationLabel;
		QLabel* _statusLabel;
	private slots:
		void reactToTextChange(QString text);
		void reactToBrowseButtonClick();
		void reactToBoundsChange(double value);
		void reactToLoadButtonClick();
		void reactToReconstructButtonClick();
		void reactToSaveButtonClick();
		void reactToRunAllButtonClick();
		void reactToStopButtonClick();
		void reactToBatchFileAction();
		void reactToLoadProgressUpdate(double percentage);
		void reactToLoadCompletion(CtVolume::CompletionStatus status);
		void reactToReconstructionProgressUpdate(double percentage, cv::Mat crossSection);
		void reactToReconstructionCompletion(cv::Mat crossSection, CtVolume::CompletionStatus status);
		void reactToSaveProgressUpdate(double percentage);
		void reactToSaveCompletion(CtVolume::CompletionStatus status);
		void askForDeletionOfIncompleteFile();
	};

}

#endif