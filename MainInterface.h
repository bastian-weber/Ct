#ifndef CT_MAININTERFACE
#define CT_MAININTERFACE

//Qt
#include <QtWidgets/QtWidgets>

namespace ct {

	class MainInterface : public QWidget{
		Q_OBJECT
	public:
		MainInterface(QWidget *parent = 0);
		~MainInterface();
		QSize sizeHint() const;
	private:

	};

}

#endif