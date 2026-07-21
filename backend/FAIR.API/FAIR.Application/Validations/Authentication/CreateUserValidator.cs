using FAIR.Application.DTOs.Identity;
using FluentValidation;

namespace FAIR.Application.Validations.Authentication
{
    public class CreateUserValidator : AbstractValidator<Register>
    {
        public CreateUserValidator()
        {
            RuleFor(x => x.FullName).NotEmpty().WithMessage("Full Name is Required.");
            RuleFor(x => x.Username).NotEmpty().WithMessage("UserName is Required.");
            RuleFor(x => x.Email).NotEmpty().WithMessage("Email is Required.")
                .EmailAddress().WithMessage("Invalid Email format.");
            RuleFor(x => x.Password).NotEmpty().WithMessage("Password is Required.")
                .MinimumLength(8).WithMessage("Password must be at least 8 characters long")
                .Matches(@"[A-Z]").WithMessage("Password must be at least one uppercase letter.")
                .Matches(@"[a-z]").WithMessage("Password must be at least one lowercase letter.")
                .Matches(@"\d").WithMessage("Password must be at least one number.")
                .Matches(@"^\w").WithMessage("Password must be at least one special character.");
            //RuleFor(x => x.ConfirmPassword).Equal(x => x.Password).WithMessage("Passwords do not match");
        }
    }
}
