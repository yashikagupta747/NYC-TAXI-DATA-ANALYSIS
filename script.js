// Smooth scrolling for navigation
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Animation on scroll
window.addEventListener('scroll', animateOnScroll);

function animateOnScroll() {
    const elements = document.querySelectorAll('.gallery-item, .insight-card');
    const windowHeight = window.innerHeight;
    
    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const animationPoint = windowHeight * 0.8;
        
        if(elementPosition < animationPoint) {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }
    });
}

// Initialize animations when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Trigger scroll event to check initial positions
    setTimeout(() => {
        window.dispatchEvent(new Event('scroll'));
    }, 100);
});