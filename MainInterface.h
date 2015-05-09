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
		void keyPressEvent(QKeyEvent * e);
	private:
		void disableAllControls();
		void startupState();
		void fileSelectedState();
		void preprocessedState();
		void reconstructedState();
		void setSinogramImage(size_t index);
		void setNextSinogramImage();
		void setPreviousSinogramImage();
		void setStatus(QString text);

		CtVolume _volume;
		bool _sinogramDisplayActive;
		bool _runAll;
		QString _savingPath;
		size_t _currentIndex;
		hb::Timer _timer;

		QVBoxLayout* _mainLayout;
		QHBoxLayout* _subLayout;
		QVBoxLayout* _leftLayout;
		QVBoxLayout* _filterLayout;
		QGroupBox* _filterGroupBox;
		QRadioButton* _ramlakRadioButton;
		QRadioButton* _shepploganRadioButton;
		QRadioButton* _hannRadioButton;
		QLineEdit* _inputFileEdit;
		QPushButton* _browseButton;
		QPushButton* _loadButton;
		QPushButton* _reconstructButton;
		QPushButton* _saveButton;
		QPushButton* _runAllButton;
		QProgressBar* _progressBar;
		hb::ImageView* _imageView;
		QLabel* _informationLabel;
		QLabel* _statusLabel;
	private slots:
		void reactToTextChange(QString text);
		void reactToBrowseButtonClick();
		void reactToLoadButtonClick();
		void reactToReconstructButtonClick();
		void reactToSaveButtonClick();
		void reactToRunAllButtonClick();
		void reactToLoadProgressUpdate(double percentage);
		void reactToLoadCompletion(CtVolume::CompletionStatus status);
		void reactToReconstructionProgressUpdate(double percentage, cv::Mat crossSection);
		void reactToReconstructionCompletion(cv::Mat crossSection, CtVolume::CompletionStatus status);
		void reactToSaveProgressUpdate(double percentage);
		void reactToSaveCompletion(CtVolume::CompletionStatus status);
	};

}

#endif