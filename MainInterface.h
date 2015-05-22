#ifndef CT_MAININTERFACE
#define CT_MAININTERFACE

#include <future>

//Qt
#include <QtWidgets/QtWidgets>

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
	protected:
		void dragEnterEvent(QDragEnterEvent* e);
		void dropEvent(QDropEvent* e);
		void keyPressEvent(QKeyEvent* e);
		void wheelEvent(QWheelEvent* e);
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
		bool _sinogramDisplayActive;
		Projection _currentProjection;
		bool _crossSectionDisplayActive;
		bool _reconstructionActive;
		bool _runAll;
		QString _savingPath;
		size_t _currentIndex;
		hb::Timer _timer;

		QVBoxLayout* _mainLayout;
		QHBoxLayout* _subLayout;
		QVBoxLayout* _leftLayout;
		QVBoxLayout* _filterLayout;
		QVBoxLayout* _boundsLayout;
		QHBoxLayout* _progressLayout;
		QHBoxLayout* _xLayout;
		QHBoxLayout* _yLayout;
		QHBoxLayout* _zLayout;
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
		QPushButton* _moreButton;
		QPushButton* _stopButton;
		QMenu* _moreMenu;
		QAction* _cmdAction;
		QProgressBar* _progressBar;
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
	};

}

#endif