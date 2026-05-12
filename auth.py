from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, PatientProfile

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form

        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        # Validation
        if not username or not email or not password:
            msg = 'All fields are required.'
            if request.is_json:
                return jsonify({'success': False, 'message': msg}), 400
            flash(msg, 'error')
            return render_template('register.html')

        if len(password) < 6:
            msg = 'Password must be at least 6 characters.'
            if request.is_json:
                return jsonify({'success': False, 'message': msg}), 400
            flash(msg, 'error')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            msg = 'Username already taken.'
            if request.is_json:
                return jsonify({'success': False, 'message': msg}), 409
            flash(msg, 'error')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            msg = 'Email already registered.'
            if request.is_json:
                return jsonify({'success': False, 'message': msg}), 409
            flash(msg, 'error')
            return render_template('register.html')

        # Create user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.flush()  # get user.id before commit

        # Create empty patient profile
        profile = PatientProfile(user_id=user.id)
        db.session.add(profile)
        db.session.commit()

        login_user(user)

        if request.is_json:
            return jsonify({'success': True, 'message': 'Account created!', 'redirect': url_for('home')})
        return redirect(url_for('home'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form

        username = data.get('username', '').strip()
        password = data.get('password', '')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=data.get('remember', False))
            if request.is_json:
                return jsonify({'success': True, 'message': 'Logged in!', 'redirect': url_for('home')})
            return redirect(url_for('home'))

        msg = 'Invalid username or password.'
        if request.is_json:
            return jsonify({'success': False, 'message': msg}), 401
        flash(msg, 'error')

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))