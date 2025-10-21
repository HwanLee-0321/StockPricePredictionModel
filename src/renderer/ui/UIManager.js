class UIManager {
    constructor() {
        this.navButtons = document.querySelectorAll('.navigation__button');
        this.contentSections = document.querySelectorAll('.content-section');
        this.initNavigation();
        this.setDefaultView();
    }

    initNavigation() {
        this.navButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.navButtons.forEach(button => button.classList.remove('navigation__button--active'));
                this.contentSections.forEach(section => section.classList.remove('content-section--active'));

                btn.classList.add('navigation__button--active');
                const targetId = btn.getAttribute('data-target');
                document.getElementById(targetId).classList.add('content-section--active');
            });
        });
    }

    setDefaultView() {
        if (this.navButtons.length > 0) {
            this.navButtons[0].classList.add('navigation__button--active');
            const defaultTargetId = this.navButtons[0].getAttribute('data-target');
            document.getElementById(defaultTargetId).classList.add('content-section--active');
        }
    }
}

export default UIManager;
