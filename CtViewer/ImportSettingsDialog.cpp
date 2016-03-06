#include "ImportSettingsDialog.h"

namespace ct {

	ImportSettingsDialog::ImportSettingsDialog(std::shared_ptr<QSettings> settings, QWidget* parent)
		: settings(settings),
		QDialog(parent) {
		this->setWindowModality(Qt::WindowModal);
		this->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
		this->setWindowFlags(this->windowFlags() & ~Qt::WindowContextHelpButtonHint);

		this->setWindowTitle(tr("Volume Parameters"));

		this->okButton = new QPushButton(tr("&Ok"), this);
		this->okButton->setDefault(true);
		QObject::connect(this->okButton, SIGNAL(clicked()), this, SLOT(accept()));

		this->cancelButton = new QPushButton(tr("&Cancel"), this);
		QObject::connect(this->cancelButton, SIGNAL(clicked()), this, SLOT(reject()));

		this->buttonLayout = new QHBoxLayout();
		this->buttonLayout->addStretch(1);
		this->buttonLayout->addWidget(okButton);
		this->buttonLayout->addWidget(cancelButton);

		this->xSpinBox = new QSpinBox;
		this->xSpinBox->setRange(1, std::numeric_limits<int>::max());
		this->xSpinBox->setSingleStep(1);
		this->ySpinBox = new QSpinBox;
		this->ySpinBox->setRange(1, std::numeric_limits<int>::max());
		this->ySpinBox->setSingleStep(1);
		this->zSpinBox = new QSpinBox;
		this->zSpinBox->setRange(1, std::numeric_limits<int>::max());
		this->zSpinBox->setSingleStep(1);
		QObject::connect(this->xSpinBox, SIGNAL(valueChanged(int)), this, SLOT(updateSize()));
		QObject::connect(this->ySpinBox, SIGNAL(valueChanged(int)), this, SLOT(updateSize()));
		QObject::connect(this->zSpinBox, SIGNAL(valueChanged(int)), this, SLOT(updateSize()));
		this->requiredSizeLabel = new QLabel("");
		this->actualSizeLabel = new QLabel("");

		this->formLayout = new QFormLayout();
		this->formLayout->addRow(tr("X size:"), this->xSpinBox);
		this->formLayout->addRow(tr("Y size:"), this->ySpinBox);
		this->formLayout->addRow(tr("Z size:"), this->zSpinBox);
		this->formLayout->addRow(tr("Actual filesize:"), this->requiredSizeLabel);
		this->formLayout->addRow(tr("Resulting filesize:"), this->actualSizeLabel);

		this->mainLayout = new QVBoxLayout;
		this->mainLayout->addLayout(this->formLayout);
		this->mainLayout->addLayout(this->buttonLayout);

		this->setLayout(this->mainLayout);
		this->layout()->setSizeConstraint(QLayout::SetFixedSize);
	}

	ImportSettingsDialog::~ImportSettingsDialog() {
		delete this->mainLayout;
		delete this->formLayout;
		delete this->buttonLayout;
		delete this->okButton;
		delete this->cancelButton;
		delete this->xSpinBox;
		delete this->ySpinBox;
		delete this->zSpinBox;
		delete this->actualSizeLabel;
		delete this->requiredSizeLabel;
	}

	int ImportSettingsDialog::execForFilesize(size_t requiredSize) {
		this->requiredSize = requiredSize;
		this->requiredSizeLabel->setText(QString::number(requiredSize).append(" bytes"));
		return this->exec();
	}

	size_t ImportSettingsDialog::getXSize() const {
		return static_cast<size_t>(this->xSpinBox->value());
	}

	size_t ImportSettingsDialog::getYSize() const {
		return static_cast<size_t>(this->ySpinBox->value());
	}

	size_t ImportSettingsDialog::getZSize() const {
		return static_cast<size_t>(this->zSpinBox->value());
	}

	void ImportSettingsDialog::showEvent(QShowEvent * e) {
		if (this->parentWidget() != 0) {
			move(this->parentWidget()->window()->frameGeometry().topLeft() + this->parentWidget()->window()->rect().center() - this->rect().center());
		}
		this->updateSize();
	}
	
	//============================================================================== PROTECTED ==============================================================================\\


	//=============================================================================== PRIVATE ===============================================================================\\


	//============================================================================ PRIVATE SLOTS =============================================================================\\

	void ImportSettingsDialog::updateSize() {
		size_t currentSize = size_t(4) * static_cast<size_t>(this->xSpinBox->value())*static_cast<size_t>(this->ySpinBox->value())*static_cast<size_t>(this->zSpinBox->value());
		this->actualSizeLabel->setText(QString::number(currentSize).append(" bytes"));
		if (currentSize == this->requiredSize) {
			this->actualSizeLabel->setStyleSheet("QLabel { }");
			this->okButton->setEnabled(true);
		} else {
			this->actualSizeLabel->setStyleSheet("QLabel { color: red; }");
			this->okButton->setEnabled(false);
		}
	}

}