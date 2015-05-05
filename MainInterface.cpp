#include "MainInterface.h"

namespace ct {

	MainInterface::MainInterface(QWidget *parent) : QWidget(parent) {

	}

	MainInterface::~MainInterface() {

	}

	QSize MainInterface::sizeHint() const {
		return QSize(1200, 700);
	}

}